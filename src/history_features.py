"""Расширенные локальные и строго исторические признаки."""

import hashlib
import logging
import re

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from data import EVENT_TYPES


LOGGER = logging.getLogger(__name__)

MISSING_COLUMNS = (
    "item_id",
    "item_category",
    "item_location",
    "seller_type",
    "search_query",
    "search_page",
    "pointer_x",
    "pointer_y",
    "user_agent",
)


def safe_ratio(numerator, denominator):
    if denominator == 0:
        return 0.0
    return numerator / denominator


def entropy(values):
    counts = pd.Series(values).value_counts()
    if counts.empty:
        return 0.0
    shares = counts / counts.sum()
    return float(-(shares * np.log2(shares)).sum())


def stable_bucket(value, size):
    digest = hashlib.blake2b(
        str(value).encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, byteorder="little") % size


def gap_bucket(seconds):
    if seconds <= 1:
        return "le_1s"
    if seconds <= 5:
        return "le_5s"
    if seconds <= 30:
        return "le_30s"
    if seconds <= 120:
        return "le_2m"
    if seconds <= 600:
        return "le_10m"
    return "gt_10m"


def longest_run(values):
    longest = 0
    current = 0
    previous = None
    for value in values:
        if value == previous:
            current += 1
        else:
            current = 1
            previous = value
        longest = max(longest, current)
    return longest


def add_numeric_summary(row, prefix, values):
    values = pd.Series(values, dtype=float).replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()
    if values.empty:
        for suffix in ("mean", "std", "median", "q10", "q90", "max"):
            row[f"{prefix}_{suffix}"] = np.nan
        return

    row[f"{prefix}_mean"] = values.mean()
    row[f"{prefix}_std"] = values.std(ddof=0)
    row[f"{prefix}_median"] = values.median()
    row[f"{prefix}_q10"] = values.quantile(0.10)
    row[f"{prefix}_q90"] = values.quantile(0.90)
    row[f"{prefix}_max"] = values.max()


def merge_feature_blocks(base, blocks):
    result = base.copy()
    for block in blocks:
        if block["cookie_id"].duplicated().any():
            raise ValueError("В блоке признаков повторяется cookie_id")
        result = result.merge(
            block,
            on="cookie_id",
            how="left",
            validate="one_to_one",
        )
    return result


class SequenceHypothesisFeatures:
    """Хешированные цепочки событий с учётом пауз между действиями."""

    @staticmethod
    def build(metadata, events, hash_size=128):
        rows = []
        groups = events.groupby("cookie_id", sort=False)

        for cookie_id, group in tqdm(
            groups,
            total=groups.ngroups,
            desc="Последовательности",
            unit="кука",
        ):
            moments = (
                group.groupby("event_ts", sort=True)["event_name"]
                .agg(lambda values: "+".join(sorted(set(values))))
            )
            tokens = moments.tolist()
            gaps = moments.index.to_series().diff().dt.total_seconds().tolist()

            transition_counts = np.zeros(hash_size, dtype=float)
            trigram_counts = np.zeros(hash_size, dtype=float)
            transitions = []

            for index in range(1, len(tokens)):
                signature = (
                    f"{tokens[index - 1]}>{tokens[index]}"
                    f"|{gap_bucket(gaps[index])}"
                )
                transitions.append(signature)
                transition_counts[stable_bucket(signature, hash_size)] += 1

            for index in range(2, len(tokens)):
                signature = (
                    f"{tokens[index - 2]}>{tokens[index - 1]}>{tokens[index]}"
                )
                trigram_counts[stable_bucket(signature, hash_size)] += 1

            transition_denominator = max(len(transitions), 1)
            trigram_denominator = max(len(tokens) - 2, 1)
            row = {
                "cookie_id": cookie_id,
                "sequence_moment_count": len(tokens),
                "sequence_token_nunique": len(set(tokens)),
                "sequence_token_entropy": entropy(tokens),
                "sequence_transition_nunique": len(set(transitions)),
                "sequence_transition_entropy": entropy(transitions),
                "sequence_transition_repeat_share": (
                    1 - safe_ratio(len(set(transitions)), len(transitions))
                ),
                "sequence_longest_same_moment_run": longest_run(tokens),
                "sequence_multi_event_moment_share": (
                    np.mean(["+" in token for token in tokens])
                    if tokens else 0
                ),
            }

            for index, value in enumerate(transition_counts):
                row[f"sequence_transition_hash_{index:03d}"] = (
                    value / transition_denominator
                )
            for index, value in enumerate(trigram_counts):
                row[f"sequence_trigram_hash_{index:03d}"] = (
                    value / trigram_denominator
                )

            event_sets = [set(token.split("+")) for token in tokens]
            funnel_pairs = (
                ("search_results_view", "item_view"),
                ("item_view", "seller_page_view"),
                ("item_view", "favorite_add"),
                ("item_view", "contact_phone_show"),
                ("item_view", "contact_chat_open"),
                ("contact_chat_open", "contact_message_sent"),
            )
            for first, second in funnel_pairs:
                name = f"sequence_{first}__{second}"
                row[name] = sum(
                    first in earlier and second in later
                    for earlier, later in zip(event_sets[:-1], event_sets[1:])
                )

            rows.append(row)

        result = metadata[["cookie_id"]].merge(
            pd.DataFrame(rows),
            on="cookie_id",
            how="left",
            validate="one_to_one",
        )
        return result.fillna(0)


class PointerDynamicsFeatures:
    """Квантование координат и геометрия движения курсора."""

    @staticmethod
    def build(metadata, events):
        rows = []
        groups = events.groupby("cookie_id", sort=False)

        for cookie_id, group in tqdm(
            groups,
            total=groups.ngroups,
            desc="Динамика курсора",
            unit="кука",
        ):
            pointer = group.dropna(subset=["pointer_x", "pointer_y"]).copy()
            row = {
                "cookie_id": cookie_id,
                "pointer_plus_count": len(pointer),
                "pointer_plus_share": safe_ratio(len(pointer), len(group)),
            }

            if pointer.empty:
                rows.append(row)
                continue

            x = pointer["pointer_x"].astype(float)
            y = pointer["pointer_y"].astype(float)
            row["pointer_plus_unique_share"] = safe_ratio(
                pointer[["pointer_x", "pointer_y"]].drop_duplicates().shape[0],
                len(pointer),
            )
            row["pointer_plus_cell_entropy_25"] = entropy(
                list(zip((x // 25).astype(int), (y // 25).astype(int)))
            )
            row["pointer_plus_cell_entropy_100"] = entropy(
                list(zip((x // 100).astype(int), (y // 100).astype(int)))
            )
            row["pointer_plus_x_last_digit_entropy"] = entropy(x.mod(10))
            row["pointer_plus_y_last_digit_entropy"] = entropy(y.mod(10))

            for step in (5, 10, 25, 50, 100):
                row[f"pointer_plus_x_multiple_{step}_share"] = np.isclose(
                    x.mod(step),
                    0,
                ).mean()
                row[f"pointer_plus_y_multiple_{step}_share"] = np.isclose(
                    y.mod(step),
                    0,
                ).mean()

            for event_name in EVENT_TYPES:
                event_pointer = pointer[pointer["event_name"] == event_name]
                prefix = f"pointer_plus_{event_name}"
                row[f"{prefix}_count"] = len(event_pointer)
                row[f"{prefix}_share"] = safe_ratio(len(event_pointer), len(pointer))
                row[f"{prefix}_x_mean"] = event_pointer["pointer_x"].mean()
                row[f"{prefix}_y_mean"] = event_pointer["pointer_y"].mean()

            ordered = pointer.drop_duplicates("event_ts", keep=False)
            if len(ordered) > 1:
                dx = ordered["pointer_x"].diff().iloc[1:].to_numpy(dtype=float)
                dy = ordered["pointer_y"].diff().iloc[1:].to_numpy(dtype=float)
                dt = (
                    ordered["event_ts"].diff().dt.total_seconds()
                    .iloc[1:].to_numpy(dtype=float)
                )
                valid = dt > 0
                dx = dx[valid]
                dy = dy[valid]
                dt = dt[valid]
                distance = np.hypot(dx, dy)
                speed = distance / dt
                angle = np.arctan2(dy, dx)
                angle_change = np.angle(
                    np.exp(1j * np.diff(angle))
                ) if len(angle) > 1 else np.array([])
                acceleration = np.diff(speed) if len(speed) > 1 else np.array([])

                add_numeric_summary(row, "pointer_plus_distance", distance)
                add_numeric_summary(row, "pointer_plus_speed", speed)
                add_numeric_summary(
                    row,
                    "pointer_plus_abs_acceleration",
                    np.abs(acceleration),
                )
                add_numeric_summary(
                    row,
                    "pointer_plus_abs_turn",
                    np.abs(angle_change),
                )
                row["pointer_plus_angle_entropy"] = entropy(
                    np.floor((angle + np.pi) / (np.pi / 8))
                )
                row["pointer_plus_zero_step_share"] = (
                    (distance == 0).mean() if len(distance) else 0
                )
                row["pointer_plus_large_jump_share"] = (
                    (distance >= 500).mean() if len(distance) else 0
                )
                row["pointer_plus_turn_share"] = (
                    (np.abs(angle_change) >= np.pi / 3).mean()
                    if len(angle_change) else 0
                )

            rows.append(row)

        return metadata[["cookie_id"]].merge(
            pd.DataFrame(rows),
            on="cookie_id",
            how="left",
            validate="one_to_one",
        )


class MissingnessFeatures:
    """Шаблон заполненности технических полей событий."""

    @staticmethod
    def build(metadata, events):
        rows = []
        groups = events.groupby("cookie_id", sort=False)

        for cookie_id, group in tqdm(
            groups,
            total=groups.ngroups,
            desc="Шаблоны пропусков",
            unit="кука",
        ):
            missing = group[list(MISSING_COLUMNS)].isna()
            pattern = missing.astype(int).astype(str).agg("".join, axis=1)
            missing_count = missing.sum(axis=1)
            row = {
                "cookie_id": cookie_id,
                "missing_pattern_nunique": pattern.nunique(),
                "missing_pattern_entropy": entropy(pattern),
                "missing_pattern_top_share": pattern.value_counts(normalize=True).iloc[0],
                "missing_fields_mean": missing_count.mean(),
                "missing_fields_std": missing_count.std(ddof=0),
                "missing_fields_max": missing_count.max(),
            }

            for column in MISSING_COLUMNS:
                row[f"missing_{column}_share"] = missing[column].mean()

            for event_name in EVENT_TYPES:
                event_group = group[group["event_name"] == event_name]
                for column in MISSING_COLUMNS:
                    name = f"missing_{column}_share__{event_name}"
                    row[name] = (
                        event_group[column].isna().mean()
                        if len(event_group) else np.nan
                    )
            rows.append(row)

        return metadata[["cookie_id"]].merge(
            pd.DataFrame(rows),
            on="cookie_id",
            how="left",
            validate="one_to_one",
        )


class TextHashFeatures:
    """Символьные фрагменты User-Agent и поисковых запросов."""

    @staticmethod
    def _char_ngrams(text, sizes):
        text = re.sub(r"\s+", " ", str(text).lower()).strip()
        for size in sizes:
            for index in range(max(len(text) - size + 1, 0)):
                yield text[index:index + size]

    @staticmethod
    def build(metadata, events, hash_size=96):
        rows = []
        groups = events.groupby("cookie_id", sort=False)

        for cookie_id, group in tqdm(
            groups,
            total=groups.ngroups,
            desc="Строки клиента и запросы",
            unit="кука",
        ):
            agents = (
                group["user_agent"].dropna().astype(str)
                .value_counts().index.tolist()
            )
            queries = (
                group["search_query"].dropna().astype(str)
                .str.lower().str.strip().tolist()
            )
            agent_document = " || ".join(agents)
            query_document = " || ".join(queries)
            agent_vector = np.zeros(hash_size, dtype=float)
            query_vector = np.zeros(hash_size, dtype=float)

            agent_ngrams = list(
                TextHashFeatures._char_ngrams(agent_document, (3, 4, 5))
            )
            query_ngrams = list(
                TextHashFeatures._char_ngrams(query_document, (3, 4))
            )
            for token in agent_ngrams:
                agent_vector[stable_bucket(token, hash_size)] += 1
            for token in query_ngrams:
                query_vector[stable_bucket(token, hash_size)] += 1

            if agent_vector.sum():
                agent_vector /= agent_vector.sum()
            if query_vector.sum():
                query_vector /= query_vector.sum()

            row = {
                "cookie_id": cookie_id,
                "ua_text_length": len(agent_document),
                "ua_text_digit_share": safe_ratio(
                    sum(char.isdigit() for char in agent_document),
                    len(agent_document),
                ),
                "ua_text_version_token_count": len(
                    re.findall(r"\d+(?:\.\d+)+", agent_document)
                ),
                "query_text_length": len(query_document),
                "query_text_word_count": len(query_document.split()),
                "query_text_digit_share": safe_ratio(
                    sum(char.isdigit() for char in query_document),
                    len(query_document),
                ),
            }
            for index, value in enumerate(agent_vector):
                row[f"ua_text_hash_{index:03d}"] = value
            for index, value in enumerate(query_vector):
                row[f"query_text_hash_{index:03d}"] = value
            rows.append(row)

        return metadata[["cookie_id"]].merge(
            pd.DataFrame(rows),
            on="cookie_id",
            how="left",
            validate="one_to_one",
        ).fillna(0)


class BatchContextFeatures:
    """Синхронная активность разных кук без использования target."""

    @staticmethod
    def build(metadata, events):
        work = events.copy()
        work["event_minute"] = work["event_ts"].dt.floor("min")
        work["event_hour"] = work["event_ts"].dt.floor("h")
        specifications = (
            ("ua_minute", ["event_minute", "user_agent"]),
            ("query_minute", ["event_minute", "search_query"]),
            ("item_hour", ["event_hour", "item_id"]),
            (
                "category_location_minute",
                ["event_minute", "item_category", "item_location"],
            ),
        )
        result = metadata[["cookie_id"]].copy()

        for prefix, columns in tqdm(
            specifications,
            desc="Синхронность кук",
            unit="группа",
        ):
            subset = work.dropna(subset=columns).copy()
            if subset.empty:
                continue
            subset["cohort_size"] = (
                subset.groupby(columns, observed=True)["cookie_id"]
                .transform("nunique")
            )
            grouped = subset.groupby("cookie_id")["cohort_size"]
            features = grouped.agg(["mean", "max"]).rename(
                columns={
                    "mean": f"batch_{prefix}_mean",
                    "max": f"batch_{prefix}_max",
                }
            )
            features[f"batch_{prefix}_q90"] = grouped.quantile(0.90)
            features[f"batch_{prefix}_share_gt_1"] = grouped.apply(
                lambda values: (values > 1).mean()
            )
            features[f"batch_{prefix}_share_gt_3"] = grouped.apply(
                lambda values: (values > 3).mean()
            )
            result = result.merge(
                features.reset_index(),
                on="cookie_id",
                how="left",
                validate="one_to_one",
            )

        return result.fillna(0)


def add_entity_columns(events):
    result = events.copy()
    result["category_location_key"] = (
        result["item_category"].astype("string")
        + "__"
        + result["item_location"].astype("string")
    )
    result["ua_category_key"] = (
        result["user_agent"].astype("string")
        + "__"
        + result["item_category"].astype("string")
    )
    return result


def aggregate_history_pairs(pairs, prefix, cookie_ids, global_rate):
    columns = [
        f"{prefix}_history_seen_share",
        f"{prefix}_history_rate_mean",
        f"{prefix}_history_rate_max",
        f"{prefix}_history_rate_q75",
        f"{prefix}_history_rate_std",
        f"{prefix}_history_rate_top3_mean",
        f"{prefix}_history_count_mean",
        f"{prefix}_history_count_max",
    ]
    result = pd.DataFrame({"cookie_id": pd.Series(cookie_ids).drop_duplicates()})

    if pairs.empty:
        for column in columns:
            result[column] = 0.0
        for column in columns:
            if "_rate_" in column:
                result[column] = global_rate
        return result

    grouped = pairs.groupby("cookie_id", sort=False)
    features = grouped.agg(
        history_seen_share=("history_count", lambda values: (values > 0).mean()),
        history_rate_mean=("history_rate", "mean"),
        history_rate_max=("history_rate", "max"),
        history_rate_q75=("history_rate", lambda values: values.quantile(0.75)),
        history_rate_std=("history_rate", lambda values: values.std(ddof=0)),
        history_count_mean=("history_count", "mean"),
        history_count_max=("history_count", "max"),
    )
    features["history_rate_top3_mean"] = grouped["history_rate"].apply(
        lambda values: values.nlargest(min(3, len(values))).mean()
    )
    features = features.rename(
        columns={column: f"{prefix}_{column}" for column in features.columns}
    )
    result = result.merge(
        features.reset_index(),
        on="cookie_id",
        how="left",
        validate="one_to_one",
    )

    rate_columns = [column for column in columns if "_rate_" in column]
    count_columns = [column for column in columns if column not in rate_columns]
    result[rate_columns] = result[rate_columns].fillna(global_rate)
    result[count_columns] = result[count_columns].fillna(0)
    return result


def build_entity_history_features(
    train_meta,
    valid_meta,
    events,
    alpha=25.0,
    entity_prefixes=None,
):
    """Строго прошлая история сущностей для train и история train для valid."""
    LOGGER.info(
        "Исторические признаки: train=%d, valid=%d, alpha=%.1f",
        len(train_meta),
        len(valid_meta),
        alpha,
    )
    required_train = {"cookie_id", "window_start_ts", "target"}
    required_valid = {"cookie_id"}
    if not required_train.issubset(train_meta.columns):
        raise ValueError("Для истории не хватает колонок train")
    if not required_valid.issubset(valid_meta.columns):
        raise ValueError("Для истории не хватает cookie_id в valid")

    train_info = train_meta[
        ["cookie_id", "window_start_ts", "target"]
    ].copy()
    train_info["date"] = pd.to_datetime(
        train_info["window_start_ts"]
    ).dt.floor("D")
    global_rate = float(train_info["target"].mean())
    train_cookie_ids = train_info["cookie_id"]
    valid_cookie_ids = valid_meta["cookie_id"]
    train_events = add_entity_columns(
        events[events["cookie_id"].isin(set(train_cookie_ids))]
    )
    valid_events = add_entity_columns(
        events[events["cookie_id"].isin(set(valid_cookie_ids))]
    )
    all_entity_columns = (
        ("item", "item_id"),
        ("query", "search_query"),
        ("user_agent", "user_agent"),
        ("category_location", "category_location_key"),
        ("ua_category", "ua_category_key"),
    )
    if entity_prefixes is None:
        entity_columns = all_entity_columns
    else:
        entity_prefixes = set(entity_prefixes)
        known_prefixes = {prefix for prefix, _ in all_entity_columns}
        unknown_prefixes = entity_prefixes - known_prefixes
        if unknown_prefixes:
            raise ValueError(
                "Неизвестные типы истории: "
                + ", ".join(sorted(unknown_prefixes))
            )
        entity_columns = tuple(
            (prefix, column)
            for prefix, column in all_entity_columns
            if prefix in entity_prefixes
        )
        if not entity_columns:
            raise ValueError("Не выбран ни один тип исторических признаков")

    train_result = pd.DataFrame({"cookie_id": train_cookie_ids.to_numpy()})
    valid_result = pd.DataFrame({"cookie_id": valid_cookie_ids.to_numpy()})

    for prefix, column in tqdm(
        entity_columns,
        desc="История сущностей",
        unit="тип",
    ):
        train_pairs = (
            train_events[["cookie_id", column]]
            .dropna()
            .drop_duplicates()
            .merge(
                train_info[["cookie_id", "date", "target"]],
                on="cookie_id",
                how="inner",
                validate="many_to_one",
            )
        )
        daily = (
            train_pairs.groupby(["date", column], observed=True)["target"]
            .agg(["sum", "count"])
            .reset_index()
            .sort_values([column, "date"])
        )
        daily["past_sum"] = (
            daily.groupby(column, observed=True)["sum"].cumsum()
            - daily["sum"]
        )
        daily["past_count"] = (
            daily.groupby(column, observed=True)["count"].cumsum()
            - daily["count"]
        )
        train_pairs = train_pairs.merge(
            daily[["date", column, "past_sum", "past_count"]],
            on=["date", column],
            how="left",
            validate="many_to_one",
        )
        train_pairs["history_count"] = train_pairs["past_count"].fillna(0)
        train_pairs["history_rate"] = (
            train_pairs["past_sum"].fillna(0) + alpha * global_rate
        ) / (train_pairs["history_count"] + alpha)

        full_stats = (
            train_pairs.groupby(column, observed=True)["target"]
            .agg(["sum", "count"])
        )
        full_stats["history_rate"] = (
            full_stats["sum"] + alpha * global_rate
        ) / (full_stats["count"] + alpha)
        valid_pairs = (
            valid_events[["cookie_id", column]]
            .dropna()
            .drop_duplicates()
            .merge(
                full_stats[["count", "history_rate"]],
                left_on=column,
                right_index=True,
                how="left",
                validate="many_to_one",
            )
            .rename(columns={"count": "history_count"})
        )
        valid_pairs["history_count"] = valid_pairs["history_count"].fillna(0)
        valid_pairs["history_rate"] = valid_pairs["history_rate"].fillna(
            global_rate
        )

        train_block = aggregate_history_pairs(
            train_pairs,
            prefix,
            train_cookie_ids,
            global_rate,
        )
        valid_block = aggregate_history_pairs(
            valid_pairs,
            prefix,
            valid_cookie_ids,
            global_rate,
        )
        train_result = train_result.merge(
            train_block,
            on="cookie_id",
            how="left",
            validate="one_to_one",
        )
        valid_result = valid_result.merge(
            valid_block,
            on="cookie_id",
            how="left",
            validate="one_to_one",
        )

    LOGGER.info(
        "Исторические признаки готовы: типов=%d, колонок=%d",
        len(entity_columns),
        train_result.shape[1] - 1,
    )
    return train_result, valid_result

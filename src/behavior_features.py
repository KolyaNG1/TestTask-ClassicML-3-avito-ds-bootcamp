"""Поведенческие признаки по событиям одной куки внутри её окна."""

import logging
import re
from collections import Counter

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from data import EVENT_TYPES, prepare_events


LOGGER = logging.getLogger(__name__)


PLATFORMS = ("web", "desktop", "android", "ios")
CONTACT_EVENTS = {"contact_phone_show", "contact_chat_open", "contact_message_sent"}


def with_progress(groups, description):
    total = groups.ngroups if hasattr(groups, "ngroups") else None
    return tqdm(groups, total=total, desc=description, unit="кука")


def ratio(numerator, denominator):
    """Возвращает 0, если знаменателя нет."""
    return numerator / denominator if denominator else 0.0


def distribution(values):
    """Краткое описание распределения непустых значений."""
    counts = pd.Series(values).dropna().value_counts()
    if counts.empty:
        return dict(unique=0, unique_share=0, repeat_share=0, top_share=0,
                    top3_share=0, entropy=0, concentration=0, singletons=0,
                    max_count=0)
    shares = counts / counts.sum()
    return dict(
        unique=len(counts),
        unique_share=len(counts) / counts.sum(),
        repeat_share=1 - len(counts) / counts.sum(),
        top_share=shares.iloc[0],
        top3_share=shares.iloc[:3].sum(),
        entropy=-(shares * np.log2(shares)).sum(),
        concentration=(shares ** 2).sum(),
        singletons=(counts == 1).sum(),
        max_count=counts.iloc[0],
    )


def longest_run(values):
    if not values:
        return 0
    longest = current = 1
    for previous, value in zip(values[:-1], values[1:]):
        current = current + 1 if value == previous else 1
        longest = max(longest, current)
    return longest


def longest_true_run(values):
    longest = current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def add_statistics(result, prefix, values):
    values = pd.Series(values).dropna().astype(float)
    result[f"{prefix}_mean"] = values.mean() if len(values) else np.nan
    result[f"{prefix}_median"] = values.median() if len(values) else np.nan
    result[f"{prefix}_min"] = values.min() if len(values) else np.nan
    result[f"{prefix}_max"] = values.max() if len(values) else np.nan
    result[f"{prefix}_std"] = values.std(ddof=0) if len(values) else np.nan
    for quantile in (0.1, 0.25, 0.75, 0.9, 0.95, 0.99):
        result[f"{prefix}_q{int(quantile * 100)}"] = values.quantile(quantile) if len(values) else np.nan


class BaselineFeatures:
    @staticmethod
    def build(metadata, events):
        counts = events.groupby("cookie_id").agg(
            n_events=("event_ts", "size"),
            item_nunique=("item_id", "nunique"),
        )
        result = metadata[["cookie_id"]].merge(counts, on="cookie_id", how="left")
        return result.fillna(0)


class BasicFeatures:
    @staticmethod
    def build(metadata, events, groups=None):
        groups = groups if groups is not None else events.groupby("cookie_id", sort=False)
        rows = []
        for cookie_id, group in with_progress(groups, "Базовые признаки"):
            first = group["event_ts"].iloc[0]
            last = group["event_ts"].iloc[-1]
            active_minutes = group["event_ts"].dt.floor("min").nunique()
            span_hours = (last - first).total_seconds() / 3600
            row = {
                "cookie_id": cookie_id,
                "n_unique_timestamps": group["event_ts"].nunique(),
                "n_event_types": group["event_name"].nunique(),
                "active_span_hours": span_hours,
                "active_minutes": active_minutes,
                "active_hours": group["event_ts"].dt.floor("h").nunique(),
                "events_per_active_minute": ratio(len(group), active_minutes),
                "events_per_span_hour": ratio(len(group), span_hours),
                "category_nunique": group["item_category"].nunique(),
                "location_nunique": group["item_location"].nunique(),
                "query_nunique": group["search_query"].nunique(),
                "search_page_nunique": group["search_page"].nunique(),
                "seller_type_nunique": group["seller_type"].nunique(),
                "max_search_page": group["search_page"].max(),
                "platform_nunique": group["platform_norm"].nunique(),
                "main_platform": group["platform_norm"].mode().iloc[0],
            }
            event_counts = group["event_name"].value_counts()
            for event_name in EVENT_TYPES:
                count = int(event_counts.get(event_name, 0))
                row[f"count_{event_name}"] = count
                row[f"share_{event_name}"] = count / len(group)
                row[f"has_{event_name}"] = int(count > 0)
            platform_counts = group["platform_norm"].value_counts()
            for platform in PLATFORMS:
                count = int(platform_counts.get(platform, 0))
                row[f"count_platform_{platform}"] = count
                row[f"share_platform_{platform}"] = count / len(group)
            rows.append(row)

        result = metadata[["cookie_id", "cookie_created_at", "window_start_ts"]].copy()
        result["cookie_age_hours"] = (
            pd.to_datetime(result["window_start_ts"])
            - pd.to_datetime(result["cookie_created_at"])
        ).dt.total_seconds() / 3600
        result["log_cookie_age"] = np.log1p(result["cookie_age_hours"].clip(lower=0))
        for hours in (6, 24, 72, 168):
            result[f"cookie_younger_than_{hours}h"] = (result["cookie_age_hours"] < hours).astype(int)
        result = result.drop(columns=["cookie_created_at", "window_start_ts"])
        return result.merge(pd.DataFrame(rows), on="cookie_id", how="left")


class TimingFeatures:
    @staticmethod
    def build(metadata, events, groups=None):
        groups = groups if groups is not None else events.groupby("cookie_id", sort=False)
        windows = metadata.set_index("cookie_id")[["window_start_ts", "window_end_ts"]]
        rows = []
        for cookie_id, group in with_progress(groups, "Временные признаки"):
            times = group["event_ts"]
            deltas = times.diff().dt.total_seconds().dropna()
            positive = deltas[deltas > 0]
            span = (times.iloc[-1] - times.iloc[0]).total_seconds()
            start = windows.loc[cookie_id, "window_start_ts"]
            end = windows.loc[cookie_id, "window_end_ts"]
            row = {
                "cookie_id": cookie_id,
                "seconds_to_first_event": (times.iloc[0] - start).total_seconds(),
                "seconds_after_last_event": (end - times.iloc[-1]).total_seconds(),
                "activity_share_of_day": span / 86400,
                "interval_unique": deltas.nunique(),
                "interval_mode_share": ratio(deltas.value_counts().iloc[0], len(deltas)) if len(deltas) else 0,
                "interval_iqr": deltas.quantile(.75) - deltas.quantile(.25) if len(deltas) else np.nan,
                "interval_mad": (deltas - deltas.median()).abs().median() if len(deltas) else np.nan,
                "interval_cv": ratio(deltas.std(ddof=0), deltas.mean()) if len(deltas) else np.nan,
                "interval_q90_to_median": ratio(deltas.quantile(.9), deltas.median()) if len(deltas) else np.nan,
                "interval_near_median_share": (
                    (deltas - deltas.median()).abs().le(2).mean() if len(deltas) else 0),
            }
            add_statistics(row, "interval", deltas)
            row["interval_positive_min"] = positive.min() if len(positive) else np.nan
            for step in (5, 10, 60):
                row[f"interval_multiple_of_{step}_share"] = (
                    (positive.mod(step) == 0).mean() if len(positive) else 0)
            for seconds in (0, 1, 2, 5, 10, 30, 60):
                row[f"interval_share_le_{seconds}s"] = (deltas <= seconds).mean() if len(deltas) else 0
            for minutes in (5, 10, 30, 60):
                row[f"interval_share_gt_{minutes}m"] = (deltas > minutes * 60).mean() if len(deltas) else 0

            minute_counts = times.dt.floor("min").value_counts()
            hour_counts = times.dt.floor("h").value_counts()
            five_minute_counts = times.dt.floor("5min").value_counts()
            row["max_events_per_minute"] = minute_counts.max()
            row["max_events_per_5min"] = five_minute_counts.max()
            row["max_events_per_hour"] = hour_counts.max()
            row["share_busiest_minute"] = minute_counts.max() / len(group)
            row["share_busiest_hour"] = hour_counts.max() / len(group)
            row["busy_minute_count"] = (minute_counts > 1).sum()

            offsets = (times - times.iloc[0]).dt.total_seconds().to_numpy()
            for part in (.25, .5, .75, .9):
                position = min(int(np.ceil(len(times) * part)) - 1, len(times) - 1)
                row[f"seconds_to_{int(part * 100)}pct_events"] = offsets[position]
            day_quarter = ((times - start).dt.total_seconds() // 21600).clip(0, 3)
            for quarter in range(4):
                row[f"events_day_quarter_{quarter}"] = int((day_quarter == quarter).sum())
            midpoint = times.iloc[0] + (times.iloc[-1] - times.iloc[0]) / 2
            first_half = (times <= midpoint).sum()
            row["second_to_first_half_ratio"] = ratio(len(times) - first_half, first_half)

            hours = times.dt.hour
            row["first_event_hour"] = hours.iloc[0]
            row["last_event_hour"] = hours.iloc[-1]
            row["main_event_hour"] = hours.mode().iloc[0]
            row["hour_entropy"] = distribution(hours)["entropy"]
            for name, low, high in (("night", 0, 6), ("morning", 6, 12),
                                    ("day", 12, 18), ("evening", 18, 24)):
                row[f"share_{name}_events"] = hours.between(low, high - 1).mean()
            row["hour_sin_mean"] = np.sin(2 * np.pi * hours / 24).mean()
            row["hour_cos_mean"] = np.cos(2 * np.pi * hours / 24).mean()

            for threshold in (15, 30):
                session_ids = deltas.reindex(group.index, fill_value=np.inf).gt(threshold * 60).cumsum()
                sessions = group.groupby(session_ids)
                sizes = sessions.size()
                session_spans = sessions["event_ts"].agg(lambda x: (x.iloc[-1] - x.iloc[0]).total_seconds() / 60)
                prefix = f"session_{threshold}m"
                row[f"{prefix}_count"] = len(sizes)
                row[f"{prefix}_size_mean"] = sizes.mean()
                row[f"{prefix}_size_median"] = sizes.median()
                row[f"{prefix}_size_max"] = sizes.max()
                row[f"{prefix}_span_mean"] = session_spans.mean()
                row[f"{prefix}_span_max"] = session_spans.max()
                row[f"{prefix}_largest_share"] = sizes.max() / len(group)
                row[f"{prefix}_single_count"] = (sizes == 1).sum()
                row[f"{prefix}_gap_mean"] = deltas[deltas > threshold * 60].mean()
                row[f"{prefix}_gap_max"] = deltas[deltas > threshold * 60].max()
                largest_session = sizes.idxmax()
                largest_group = group.loc[session_ids == largest_session]
                row[f"{prefix}_largest_item_nunique"] = largest_group["item_id"].nunique()
                row[f"{prefix}_largest_type_nunique"] = largest_group["event_name"].nunique()
                row[f"{prefix}_types_mean"] = sessions["event_name"].nunique().mean()
            rows.append(row)
        return pd.DataFrame(rows)


class BehaviorFeatures:
    @staticmethod
    def build(metadata, events, groups=None):
        groups = groups if groups is not None else events.groupby("cookie_id", sort=False)
        rows = []
        fields = {
            "item": "item_id", "category": "item_category",
            "location": "item_location", "query": "search_query",
        }
        for cookie_id, group in with_progress(groups, "Структура поведения"):
            row = {"cookie_id": cookie_id}
            for name, column in fields.items():
                stats = distribution(group[column])
                for stat_name, value in stats.items():
                    row[f"{name}_{stat_name}"] = value

            row["repeated_item_count"] = int((group["item_id"].value_counts() > 1).sum())
            ordered = group.drop_duplicates("event_ts", keep=False)
            for name, column in (("item", "item_id"), ("category", "item_category"),
                                 ("location", "item_location"), ("query", "search_query")):
                values = ordered[column].dropna().tolist()
                row[f"{name}_same_next_share"] = ratio(
                    sum(a == b for a, b in zip(values[:-1], values[1:])), len(values) - 1)
                row[f"{name}_switch_count"] = sum(a != b for a, b in zip(values[:-1], values[1:]))
                row[f"{name}_longest_run"] = longest_run(values)
                row[f"{name}_return_count"] = sum(
                    value in values[:index - 1] and value != values[index - 1]
                    for index, value in enumerate(values) if index > 0
                )

            session_ids = group["event_ts"].diff().dt.total_seconds().gt(1800).fillna(True).cumsum()
            item_sessions = group.assign(session_id=session_ids).dropna(subset=["item_id"])
            row["items_in_multiple_sessions"] = int(
                (item_sessions.groupby("item_id")["session_id"].nunique() > 1).sum())

            seller = group["seller_type"].value_counts()
            for kind in ("private", "pro"):
                row[f"seller_{kind}_share"] = ratio(seller.get(kind, 0), seller.sum())
                row[f"seller_{kind}_views"] = int(((group["seller_type"] == kind)
                                                     & (group["event_name"] == "item_view")).sum())
                row[f"seller_{kind}_contacts"] = int(((group["seller_type"] == kind)
                                                        & group["event_name"].isin(CONTACT_EVENTS)).sum())
            row["seller_both_types"] = int(len(seller) == 2)
            row["seller_switch_count"] = sum(
                a != b for a, b in zip(ordered["seller_type"].dropna().tolist()[:-1],
                                        ordered["seller_type"].dropna().tolist()[1:]))

            search = group[group["event_name"] == "search_results_view"].copy()
            pages = search["search_page"].dropna()
            row["search_page_mean"] = pages.mean()
            row["search_page_median"] = pages.median()
            row["search_page_q90"] = pages.quantile(.9)
            for limit in (1, 3, 5, 10, 20):
                row[f"search_page_gt_{limit}_share"] = (pages > limit).mean() if len(pages) else 0
            page_pairs = search.drop_duplicates("event_ts", keep=False)["search_page"].dropna().tolist()
            row["search_next_page_count"] = sum(b == a + 1 for a, b in zip(page_pairs[:-1], page_pairs[1:]))
            row["search_reset_page_count"] = sum(b == 1 and a > 1 for a, b in zip(page_pairs[:-1], page_pairs[1:]))
            row["search_page_longest_run"] = longest_true_run([
                b == a + 1 for a, b in zip(page_pairs[:-1], page_pairs[1:])])
            row["search_pages_per_query"] = ratio(pages.nunique(), search["search_query"].nunique())
            pages_per_query = search.groupby("search_query")["search_page"].nunique()
            row["search_pages_per_query_mean"] = pages_per_query.mean()
            row["search_pages_per_query_max"] = pages_per_query.max()
            row["search_max_depth_per_query"] = search.groupby("search_query")["search_page"].max().max()

            queries = search["search_query"].dropna().astype(str).str.strip().str.lower()
            row["query_length_mean"] = queries.str.len().mean()
            row["query_word_count_mean"] = queries.str.split().str.len().mean()
            row["query_length_max"] = queries.str.len().max()
            row["query_digit_count_mean"] = queries.str.count(r"\d").mean()
            row["query_digit_share_mean"] = queries.apply(
                lambda text: ratio(sum(char.isdigit() for char in text), len(text))).mean()
            row["query_with_digit_share"] = queries.str.contains(r"\d", regex=True).mean() if len(queries) else 0
            row["query_unique_words"] = len(set(" ".join(queries).split()))
            query_list = queries.tolist()
            row["query_small_change_count"] = sum(
                0 < len(set(a.split()) ^ set(b.split())) <= 2
                for a, b in zip(query_list[:-1], query_list[1:]))

            counts = group["event_name"].value_counts()
            n_search = counts.get("search_results_view", 0)
            n_view = counts.get("item_view", 0)
            n_chat = counts.get("contact_chat_open", 0)
            n_contacts = sum(counts.get(name, 0) for name in CONTACT_EVENTS)
            for name, numerator, denominator in (
                ("views_per_search", n_view, n_search),
                ("photos_per_view", counts.get("photo_swipe", 0), n_view),
                ("seller_pages_per_view", counts.get("seller_page_view", 0), n_view),
                ("contacts_per_view", n_contacts, n_view),
                ("favorites_per_view", counts.get("favorite_add", 0), n_view),
                ("messages_per_chat", counts.get("contact_message_sent", 0), n_chat),
                ("contacts_per_item", n_contacts, group["item_id"].nunique()),
            ):
                row[name] = ratio(numerator, denominator)

            # У событий одной секунды порядок не определён: переходы между ними не считаем.
            moments = group.groupby("event_ts")["event_name"].apply(lambda values: set(values))
            transitions = Counter()
            for earlier, later in zip(moments.iloc[:-1], moments.iloc[1:]):
                for first in earlier:
                    for second in later:
                        transitions[(first, second)] += 1
            transition_total = sum(transitions.values())
            row["transition_unique"] = len(transitions)
            row["transition_change_count"] = sum(
                count for (first, second), count in transitions.items() if first != second)
            row["transition_same_share"] = ratio(
                sum(count for (first, second), count in transitions.items() if first == second),
                transition_total)
            row["transition_entropy"] = distribution(list(transitions.elements()))["entropy"] if transitions else 0
            single_event_moments = [next(iter(names)) for names in moments if len(names) == 1]
            row["event_longest_run"] = longest_run(single_event_moments)
            row["event_return_aba_count"] = sum(
                first == third and first != second
                for first, second, third in zip(single_event_moments[:-2],
                                                 single_event_moments[1:-1],
                                                 single_event_moments[2:]))
            for first in EVENT_TYPES:
                for second in EVENT_TYPES:
                    count = transitions[(first, second)]
                    row[f"transition_{first}__{second}"] = count
                    row[f"transition_share_{first}__{second}"] = ratio(count, transition_total)

            # Воронку считаем по событиям строго позже исходного, в пределах пяти минут.
            search_without_view = 0
            view_without_action = 0
            event_names = group["event_name"].to_numpy()
            event_times = group["event_ts"].to_numpy(dtype="datetime64[ns]")
            for position, event_name in enumerate(event_names):
                if event_name not in {"search_results_view", "item_view"}:
                    continue
                end = np.searchsorted(event_times, event_times[position] + np.timedelta64(5, "m"), side="right")
                later_names = event_names[position + 1:end]
                later_times = event_times[position + 1:end]
                later_names = later_names[later_times > event_times[position]]
                if event_name == "search_results_view":
                    search_without_view += int(not (later_names == "item_view").any())
                else:
                    followup = {"photo_swipe", "seller_page_view", "favorite_add"} | CONTACT_EVENTS
                    view_without_action += int(not np.isin(later_names, list(followup)).any())
            row["search_without_view_share"] = ratio(search_without_view, n_search)
            row["view_without_action_share"] = ratio(view_without_action, n_view)

            moment_items = list(moments.items())
            for first, second in (("search_results_view", "item_view"),
                                  ("item_view", "photo_swipe"),
                                  ("item_view", "seller_page_view"),
                                  ("item_view", "contact_phone_show"),
                                  ("contact_chat_open", "contact_message_sent")):
                gaps = []
                for (time_a, names_a), (time_b, names_b) in zip(moment_items[:-1], moment_items[1:]):
                    if first in names_a and second in names_b:
                        gaps.append((time_b - time_a).total_seconds())
                row[f"gap_{first}__{second}_median"] = np.median(gaps) if gaps else np.nan

            row["extra_duplicate_count"] = int(group["extra_duplicate"].sum())
            row["extra_duplicate_share"] = group["extra_duplicate"].mean()
            duplicate_sizes = group.groupby(list(RAW_EVENT_COLUMNS), dropna=False).size()
            row["duplicate_group_count"] = int((duplicate_sizes > 1).sum())
            row["duplicate_group_max"] = int(duplicate_sizes.max())
            for event_name in EVENT_TYPES:
                row[f"duplicate_{event_name}"] = int(group.loc[
                    group["event_name"] == event_name, "extra_duplicate"].sum())
            timestamp_counts = group["event_ts"].value_counts()
            row["multi_event_second_count"] = int((timestamp_counts > 1).sum())
            row["max_events_same_second"] = int(timestamp_counts.max())
            row["multi_event_second_share"] = ratio(timestamp_counts[timestamp_counts > 1].sum(), len(group))
            for column in ("item_id", "item_category", "item_location", "seller_type",
                           "search_query", "search_page", "pointer_x", "pointer_y"):
                row[f"filled_{column}_share"] = group[column].notna().mean()
            row["missing_pattern_count"] = group[
                ["item_id", "item_category", "item_location", "seller_type",
                 "search_query", "search_page", "pointer_x", "pointer_y"]
            ].notna().drop_duplicates().shape[0]
            rows.append(row)
        return pd.DataFrame(rows)


def user_agent_family(value):
    value = str(value).lower()
    for marker, family in (
        ("headlesschrome", "headless_chrome"), ("scrapy", "scrapy"),
        ("node-fetch", "node_fetch"), ("curl/", "curl"),
        ("urllib3", "urllib3"), ("python-requests", "requests"),
        ("go-http-client", "go_http"), ("yabrowser", "yandex_browser"),
        ("firefox", "firefox"), ("mobile safari", "mobile_browser"),
        ("chrome", "chrome"), ("avito/", "avito_app"),
    ):
        if marker in value:
            return family
    return "other"


def operating_system(value):
    value = str(value).lower()
    for marker, name in (("android", "android"), ("iphone", "ios"),
                         ("windows", "windows"), ("macintosh", "macos"),
                         ("linux", "linux")):
        if marker in value:
            return name
    return "other"


def safe_name(value):
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def longest_fast_run(deltas, threshold):
    """Максимальное число событий подряд с паузой не больше threshold."""
    if len(deltas) == 0:
        return 1
    return longest_true_run((deltas <= threshold).tolist()) + 1


class FullFeatures:
    @staticmethod
    def build(metadata, events, vocabulary_cookie_ids, groups=None):
        groups = groups if groups is not None else events.groupby("cookie_id", sort=False)
        rows = []
        vocabulary_cookie_ids = set(vocabulary_cookie_ids)
        if not vocabulary_cookie_ids:
            raise ValueError("Не переданы обучающие куки для словарей признаков")
        vocabulary_events = events.loc[events["cookie_id"].isin(vocabulary_cookie_ids)]
        categories = sorted(vocabulary_events["item_category"].dropna().unique())
        locations = sorted(vocabulary_events["item_location"].dropna().unique())

        for cookie_id, group in with_progress(groups, "Полный набор"):
            agent_values = group["user_agent"].fillna("missing").astype(str)
            agent_counts = agent_values.value_counts()
            main_agent = agent_counts.index[0]
            family = user_agent_family(main_agent)
            os_name = operating_system(main_agent)
            row = {
                "cookie_id": cookie_id,
                "main_user_agent": main_agent,
                "ua_family": family,
                "ua_os": os_name,
                "main_category": group["item_category"].mode().iloc[0] if group["item_category"].notna().any() else "missing",
                "main_location": group["item_location"].mode().iloc[0] if group["item_location"].notna().any() else "missing",
                "main_query": group["search_query"].mode().iloc[0] if group["search_query"].notna().any() else "missing",
                "main_seller_type": group["seller_type"].mode().iloc[0] if group["seller_type"].notna().any() else "missing",
                "ua_nunique": agent_values.nunique(),
                "ua_family_nunique": agent_values.map(user_agent_family).nunique(),
                "ua_main_share": agent_counts.iloc[0] / len(group),
                "ua_length": len(main_agent),
            }
            unique_moments = group.drop_duplicates("event_ts", keep=False)
            ua_values = unique_moments["user_agent"].fillna("missing").astype(str).tolist()
            row["ua_switch_count"] = sum(a != b for a, b in zip(ua_values[:-1], ua_values[1:]))
            for marker in ("headlesschrome", "scrapy", "curl/", "node-fetch", "urllib3",
                           "python-requests", "go-http-client"):
                row[f"ua_has_{safe_name(marker)}"] = int(marker in main_agent.lower())
            row["ua_is_program"] = int(family in {
                "headless_chrome", "scrapy", "curl", "node_fetch", "urllib3", "requests", "go_http"})
            row["ua_is_mobile"] = int(os_name in {"android", "ios"})
            row["ua_platform_mismatch"] = int(
                (os_name in {"android", "ios"} and group["platform_norm"].eq("desktop").any())
                or (os_name in {"windows", "macos", "linux"} and group["platform_norm"].isin(["android", "ios"]).any()))
            version = re.search(r"(?:Chrome|Firefox|Avito)/([0-9]+)", main_agent)
            row["ua_major_version"] = int(version.group(1)) if version else np.nan
            row["category_platform"] = f"{row['main_category']}__{group['platform_norm'].mode().iloc[0]}"
            row["category_seller"] = f"{row['main_category']}__{row['main_seller_type']}"

            pointer = group.dropna(subset=["pointer_x", "pointer_y"])
            web_events = group["platform_norm"].isin(["web", "desktop"]).sum()
            row["pointer_count"] = len(pointer)
            row["pointer_share"] = len(pointer) / len(group)
            row["pointer_web_share"] = ratio(len(pointer), web_events)
            row["pointer_unique_points"] = pointer[["pointer_x", "pointer_y"]].drop_duplicates().shape[0]
            row["pointer_unique_x"] = pointer["pointer_x"].nunique()
            row["pointer_unique_y"] = pointer["pointer_y"].nunique()
            row["pointer_zero_share"] = ((pointer["pointer_x"] == 0) | (pointer["pointer_y"] == 0)).mean() if len(pointer) else 0
            row["pointer_center_share"] = (
                pointer["pointer_x"].between(480, 1440)
                & pointer["pointer_y"].between(270, 810)).mean() if len(pointer) else 0
            row["pointer_edge_share"] = (
                (pointer["pointer_x"] < 100) | (pointer["pointer_x"] > 1820)
                | (pointer["pointer_y"] < 100) | (pointer["pointer_y"] > 980)).mean() if len(pointer) else 0
            row["pointer_point_repeat_share"] = 1 - ratio(row["pointer_unique_points"], len(pointer))
            for axis in ("x", "y"):
                values = pointer[f"pointer_{axis}"]
                row[f"pointer_{axis}_mean"] = values.mean()
                row[f"pointer_{axis}_median"] = values.median()
                row[f"pointer_{axis}_std"] = values.std(ddof=0)
                row[f"pointer_{axis}_range"] = values.max() - values.min() if len(values) else np.nan
            row["pointer_box_area"] = row["pointer_x_range"] * row["pointer_y_range"]
            # Координаты с одной и той же отметкой времени не задают маршрут.
            ordered_pointer = pointer.drop_duplicates("event_ts", keep=False)
            dx = ordered_pointer["pointer_x"].diff()
            dy = ordered_pointer["pointer_y"].diff()
            distance = np.sqrt(dx ** 2 + dy ** 2).dropna()
            row["pointer_move_mean"] = distance.mean()
            row["pointer_move_median"] = distance.median()
            row["pointer_move_max"] = distance.max()
            row["pointer_path_length"] = distance.sum()
            row["pointer_still_share"] = (distance == 0).mean() if len(distance) else 0
            row["pointer_horizontal_share"] = ((dy == 0) & (dx != 0)).mean() if len(ordered_pointer) > 1 else 0
            row["pointer_vertical_share"] = ((dx == 0) & (dy != 0)).mean() if len(ordered_pointer) > 1 else 0
            direction = np.sign(dx.fillna(0)) + 3 * np.sign(dy.fillna(0))
            row["pointer_direction_change_count"] = int(
                ((direction != direction.shift()) & (direction != 0)).sum()) if len(direction) else 0
            if len(ordered_pointer) > 1:
                direct = np.hypot(ordered_pointer["pointer_x"].iloc[-1] - ordered_pointer["pointer_x"].iloc[0],
                                  ordered_pointer["pointer_y"].iloc[-1] - ordered_pointer["pointer_y"].iloc[0])
                row["pointer_straightness"] = ratio(direct, distance.sum())
                speed = distance / ordered_pointer["event_ts"].diff().dt.total_seconds().dropna()
                row["pointer_speed_mean"] = speed.mean()
                row["pointer_speed_max"] = speed.max()
            else:
                row["pointer_straightness"] = np.nan
                row["pointer_speed_mean"] = np.nan
                row["pointer_speed_max"] = np.nan
            rows.append(row)

        result = pd.DataFrame(rows)
        event_counts = events.groupby("cookie_id").size()

        for name, column, values in (("category", "item_category", categories),
                                     ("location", "item_location", locations)):
            counts = (
                events.loc[events[column].isin(values)]
                .groupby(["cookie_id", column], observed=True)
                .size()
                .unstack(fill_value=0)
                .reindex(columns=values, fill_value=0)
            )
            count_names = [f"count_{name}_{safe_name(value)}" for value in values]
            share_names = [f"share_{name}_{safe_name(value)}" for value in values]
            shares = counts.div(event_counts.reindex(counts.index), axis=0)
            counts.columns = count_names
            shares.columns = share_names
            wide = pd.concat([counts, shares], axis=1).reset_index()
            result = result.merge(wide, on="cookie_id", how="left")
            result[count_names + share_names] = result[count_names + share_names].fillna(0)

        platform_columns = pd.MultiIndex.from_product(
            [PLATFORMS, EVENT_TYPES], names=["platform_norm", "event_name"])
        platform_counts = (
            events.groupby(["cookie_id", "platform_norm", "event_name"], observed=True)
            .size()
            .unstack(["platform_norm", "event_name"], fill_value=0)
            .reindex(columns=platform_columns, fill_value=0)
        )
        platform_counts.columns = [
            f"count_{platform}_{event_name}"
            for platform, event_name in platform_counts.columns
        ]
        result = result.merge(platform_counts.reset_index(), on="cookie_id", how="left")
        result[platform_counts.columns] = result[platform_counts.columns].fillna(0)
        return result


class AdvancedFeatures:
    """Признаки с упором на регулярность, скорость обхода и профиль поиска."""

    @staticmethod
    def build(metadata, events, vocabulary_cookie_ids, groups=None):
        groups = groups if groups is not None else events.groupby("cookie_id", sort=False)
        vocabulary_cookie_ids = set(vocabulary_cookie_ids)
        vocabulary_events = events.loc[events["cookie_id"].isin(vocabulary_cookie_ids)]
        queries = sorted(vocabulary_events["search_query"].dropna().astype(str).unique())
        rows = []

        for cookie_id, group in with_progress(groups, "Усиленные признаки"):
            times = group["event_ts"]
            deltas = times.diff().dt.total_seconds().dropna()
            positive = deltas[deltas > 0]
            positive_stats = distribution(positive)
            span_minutes = max((times.iloc[-1] - times.iloc[0]).total_seconds() / 60, 1 / 60)
            row = {"cookie_id": cookie_id}

            second_counts = times.dt.floor("s").value_counts()
            seconds = times.dt.second
            minutes = times.dt.minute
            row["events_per_span_minute"] = len(group) / span_minutes
            row["unique_items_per_span_minute"] = group["item_id"].nunique() / span_minutes
            row["unique_queries_per_span_minute"] = group["search_query"].nunique() / span_minutes
            row["max_events_same_timestamp"] = int(second_counts.max())
            row["same_timestamp_event_share"] = second_counts[second_counts > 1].sum() / len(group)
            row["event_second_zero_share"] = (seconds == 0).mean()
            row["event_second_multiple_5_share"] = (seconds.mod(5) == 0).mean()
            row["event_second_nunique"] = seconds.nunique()
            row["event_second_entropy"] = distribution(seconds)["entropy"]
            row["event_minute_nunique"] = minutes.nunique()
            row["event_minute_entropy"] = distribution(minutes)["entropy"]
            row["positive_interval_nunique_share"] = ratio(positive.nunique(), len(positive))
            row["positive_interval_top_share"] = positive_stats["top_share"]
            row["positive_interval_entropy"] = positive_stats["entropy"]

            for threshold in (1, 2, 5, 10, 30, 60):
                row[f"fast_run_le_{threshold}s"] = longest_fast_run(deltas, threshold)

            event_times_by_name = {
                name: values for name, values in group.groupby("event_name")["event_ts"]
            }
            for event_name in EVENT_TYPES:
                event_times = event_times_by_name.get(event_name, pd.Series(dtype="datetime64[ns]"))
                event_deltas = event_times.diff().dt.total_seconds().dropna()
                prefix = f"{event_name}_timing"
                row[f"{prefix}_span_minutes"] = (
                    (event_times.iloc[-1] - event_times.iloc[0]).total_seconds() / 60
                    if len(event_times) > 1 else 0
                )
                row[f"{prefix}_gap_median"] = event_deltas.median()
                row[f"{prefix}_gap_q10"] = event_deltas.quantile(0.1)
                row[f"{prefix}_gap_q90"] = event_deltas.quantile(0.9)
                row[f"{prefix}_gap_cv"] = ratio(
                    event_deltas.std(ddof=0), event_deltas.mean()
                ) if len(event_deltas) else np.nan

            item_events = group.dropna(subset=["item_id"])
            events_per_item = item_events.groupby("item_id").size()
            item_views = group.loc[group["event_name"] == "item_view"].dropna(subset=["item_id"])
            views_per_item = item_views.groupby("item_id").size()
            row["events_on_item_mean"] = events_per_item.mean()
            row["events_on_item_max"] = events_per_item.max()
            row["single_event_item_share"] = (events_per_item == 1).mean() if len(events_per_item) else 0
            row["views_per_item_mean"] = views_per_item.mean()
            row["views_per_item_max"] = views_per_item.max()
            row["revisited_item_share"] = (views_per_item > 1).mean() if len(views_per_item) else 0

            search = group.loc[group["event_name"] == "search_results_view"].copy()
            search_queries = search["search_query"].tolist()
            query_page = search.dropna(subset=["search_query", "search_page"])
            query_page_counts = query_page.groupby(["search_query", "search_page"]).size()
            row["query_page_pair_nunique"] = len(query_page_counts)
            row["query_page_repeat_share"] = (
                1 - ratio(len(query_page_counts), len(query_page)) if len(query_page) else 0
            )
            row["same_query_consecutive_share"] = ratio(
                sum(a == b for a, b in zip(search_queries[:-1], search_queries[1:])),
                len(search) - 1,
            )
            rows.append(row)

        result = pd.DataFrame(rows)
        event_counts = events.groupby("cookie_id").size()

        query_counts = (
            events.loc[events["search_query"].notna()]
            .assign(search_query=lambda frame: frame["search_query"].astype(str))
            .loc[lambda frame: frame["search_query"].isin(queries)]
            .groupby(["cookie_id", "search_query"], observed=True)
            .size()
            .unstack(fill_value=0)
            .reindex(columns=queries, fill_value=0)
        )
        # В текстах запросов есть кириллица и знаки пунктуации. Порядковый номер
        # словаря надёжнее длинных и потенциально совпадающих названий колонок.
        query_count_names = [f"count_query_{index:03d}" for index in range(len(queries))]
        query_share_names = [name.replace("count_", "share_", 1) for name in query_count_names]
        query_shares = query_counts.div(event_counts.reindex(query_counts.index), axis=0)
        query_counts.columns = query_count_names
        query_shares.columns = query_share_names
        query_wide = pd.concat([query_counts, query_shares], axis=1).reset_index()
        result = result.merge(query_wide, on="cookie_id", how="left")
        result[query_count_names + query_share_names] = (
            result[query_count_names + query_share_names].fillna(0)
        )

        ua_profile = events[["cookie_id", "user_agent"]].copy()
        ua_profile["ua_family"] = ua_profile["user_agent"].map(user_agent_family)
        ua_profile["ua_os"] = ua_profile["user_agent"].map(operating_system)
        for column in ("ua_family", "ua_os"):
            counts = ua_profile.groupby(["cookie_id", column], observed=True).size().unstack(fill_value=0)
            count_names = [f"count_{column}_{safe_name(value)}" for value in counts.columns]
            share_names = [name.replace("count_", "share_", 1) for name in count_names]
            shares = counts.div(event_counts.reindex(counts.index), axis=0)
            counts.columns = count_names
            shares.columns = share_names
            wide = pd.concat([counts, shares], axis=1).reset_index()
            result = result.merge(wide, on="cookie_id", how="left")
            result[count_names + share_names] = result[count_names + share_names].fillna(0)

        seller_event = (
            events.dropna(subset=["seller_type"])
            .groupby(["cookie_id", "seller_type", "event_name"], observed=True)
            .size()
            .unstack(["seller_type", "event_name"], fill_value=0)
        )
        seller_event.columns = [
            f"count_seller_{safe_name(seller)}_{event_name}"
            for seller, event_name in seller_event.columns
        ]
        result = result.merge(seller_event.reset_index(), on="cookie_id", how="left")
        result[seller_event.columns] = result[seller_event.columns].fillna(0)
        return result


def build_feature_sets(metadata, events, vocabulary_cookie_ids, events_are_prepared=False):
    """Накопительные наборы. Каждый следующий содержит предыдущий."""
    LOGGER.info("Начинаю расчёт поведенческих признаков для %d кук", len(metadata))
    prepared = events if events_are_prepared else prepare_events(metadata, events)
    # Один объект группировки переиспользуется всеми блоками. Это не копирует
    # тысячи маленьких таблиц в память и не пересчитывает индексы групп заново.
    groups = prepared.groupby("cookie_id", sort=False)
    baseline = BaselineFeatures.build(metadata, prepared)
    basic = baseline.merge(
        BasicFeatures.build(metadata, prepared, groups), on="cookie_id", how="left")
    timing = basic.merge(
        TimingFeatures.build(metadata, prepared, groups), on="cookie_id", how="left")
    behavior = timing.merge(
        BehaviorFeatures.build(metadata, prepared, groups), on="cookie_id", how="left")
    full = behavior.merge(
        FullFeatures.build(metadata, prepared, vocabulary_cookie_ids, groups),
        on="cookie_id", how="left")
    for column in ("n_events", "item_nunique", "category_nunique", "location_nunique",
                   "query_nunique", "active_minutes", "extra_duplicate_count"):
        full[f"log_{column}"] = np.log1p(full[column].clip(lower=0))
    for name, numerator, denominator in (
        ("events_per_item", "n_events", "item_nunique"),
        ("events_per_category", "n_events", "category_nunique"),
        ("events_per_location", "n_events", "location_nunique"),
        ("events_per_query", "n_events", "query_nunique"),
        ("items_per_category", "item_nunique", "category_nunique"),
        ("items_per_location", "item_nunique", "location_nunique"),
        ("items_per_query", "item_nunique", "query_nunique"),
        ("contacts_per_session", "count_contact_phone_show", "session_30m_count"),
        ("duplicate_per_active_minute", "extra_duplicate_count", "active_minutes"),
    ):
        full[name] = full[numerator].div(full[denominator].replace(0, np.nan)).fillna(0)

    advanced = full.merge(
        AdvancedFeatures.build(metadata, prepared, vocabulary_cookie_ids, groups),
        on="cookie_id", how="left",
    )

    for frame in (baseline, basic, timing, behavior, full, advanced):
        if len(frame) != len(metadata) or frame["cookie_id"].duplicated().any():
            raise ValueError("После сборки признаки не соответствуют списку кук")

    feature_sets = {
        "minimal_baseline": baseline,
        "basic_behavior": basic,
        "temporal_behavior": timing,
        "behavior_structure": behavior,
        "full_behavior": full,
        "extended_behavior": advanced,
    }
    LOGGER.info(
        "Поведенческие наборы готовы: %s",
        ", ".join(
            f"{name}={frame.shape[1] - 1}"
            for name, frame in feature_sets.items()
        ),
    )
    return feature_sets

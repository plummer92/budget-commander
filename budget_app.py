import sqlite3
from datetime import datetime, date, timedelta
from calendar import monthrange

import pandas as pd
import streamlit as st
import random

DB_PATH = "budget.db"

# =================== CATEGORY SETUP ===================

WEEKLY_CATEGORIES = [
    "Groceries",
    "Eating Out",
    "Gas & Transportation",
    "Pets",
    "Medical & Rx",
    "Shopping / Misc",
    "Household",
    "Entertainment",
    "Savings",
]


MONTHLY_CATEGORIES = [
    "Housing (Mortgage)",
    "Bills & Utilities",
    "Subscriptions",
    "Insurance",
    "Debt Payoff (Jenius Loan)",
    "Investing",
    "Home Projects",
    "Travel Fund",
]

# =================== DB INIT ===================

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Transactions table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tx_date TEXT NOT NULL,
            description TEXT,
            category TEXT NOT NULL,
            amount REAL NOT NULL,
            tx_type TEXT NOT NULL,  -- 'income' or 'expense'
            account TEXT
        )
        """
    )

        # Pay period savings + extra debt settings
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS period_adjustments (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            savings REAL NOT NULL DEFAULT 0,
            extra_debt REAL NOT NULL DEFAULT 0
        )
        """
    )

    # Ensure row exists
    cur.execute("SELECT COUNT(*) FROM period_adjustments")
    if cur.fetchone()[0] == 0:
        cur.execute("INSERT INTO period_adjustments (id, savings, extra_debt) VALUES (1, 0, 0)")


    # Gamification core stats
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS gamification (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            total_xp INTEGER NOT NULL DEFAULT 0,
            streak INTEGER NOT NULL DEFAULT 0,
            best_streak INTEGER NOT NULL DEFAULT 0,
            last_activity_date TEXT
        )
        """
    )

    cur.execute("SELECT COUNT(*) FROM gamification")
    if cur.fetchone()[0] == 0:
        cur.execute(
            "INSERT INTO gamification (id, total_xp, streak, best_streak, last_activity_date) "
            "VALUES (1, 0, 0, 0, NULL)"
        )

    # Badges
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS badges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            earned_date TEXT NOT NULL
        )
        """
    )

    # Pay period settings (anchored to 2025-11-14, 14-day cycle by default)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS pay_period_settings (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            reference_payday TEXT NOT NULL,
            period_days INTEGER NOT NULL,
            pot_budget REAL NOT NULL DEFAULT 0
        )
        """
    )

    cur.execute("SELECT COUNT(*) FROM pay_period_settings")
    if cur.fetchone()[0] == 0:
        # Default reference payday + cycle
        cur.execute(
            "INSERT INTO pay_period_settings (id, reference_payday, period_days, pot_budget) "
            "VALUES (1, ?, ?, ?)",
            ("2025-11-14", 14, 0.0),
        )

    # Category auto-rules (ALWAYS create this table)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS category_rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern TEXT NOT NULL UNIQUE,
            category TEXT NOT NULL
        )
        """
    )

        # Bills / fixed expenses
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS bills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            amount REAL NOT NULL,
            frequency TEXT NOT NULL DEFAULT 'monthly'
        )
        """
    )


    conn.commit()
    conn.close()

# =================== TRANSACTIONS ===================

def add_transaction(tx_date, description, category, amount, tx_type, account):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO transactions (tx_date, description, category, amount, tx_type, account)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (tx_date, description, category, amount, tx_type, account),
    )
    conn.commit()
    conn.close()
    update_gamification_on_activity()


def load_transactions(start_date=None, end_date=None):
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT id, tx_date, description, category, amount, tx_type, account FROM transactions"
    params = []
    if start_date and end_date:
        query += " WHERE date(tx_date) BETWEEN date(?) AND date(?)"
        params.extend([start_date, end_date])
    elif start_date:
        query += " WHERE date(tx_date) >= date(?)"
        params.append(start_date)
    elif end_date:
        query += " WHERE date(tx_date) <= date(?)"
        params.append(end_date)

    query += " ORDER BY date(tx_date) DESC, id DESC"
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    if not df.empty:
        df["tx_date"] = pd.to_datetime(df["tx_date"]).dt.date
    return df

def update_transaction_categories(tx_ids, new_category: str):
    """Bulk update category for a list of transaction IDs."""
    if not tx_ids:
        return
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    placeholders = ",".join("?" * len(tx_ids))
    query = f"UPDATE transactions SET category = ? WHERE id IN ({placeholders})"
    cur.execute(query, [new_category, *tx_ids])
    conn.commit()
    conn.close()

def add_or_update_rule(pattern: str, category: str):
    """
    Save or update a rule: if description contains `pattern`, set category to `category`.
    Pattern is stored uppercased and matched case-insensitively.
    """
    pattern = pattern.strip().upper()
    if not pattern:
        return
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO category_rules (pattern, category)
        VALUES (?, ?)
        ON CONFLICT(pattern) DO UPDATE SET category = excluded.category
        """,
        (pattern, category),
    )
    conn.commit()
    conn.close()


def get_rules() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT pattern, category FROM category_rules", conn)
    conn.close()
    return df


def apply_rules_to_other() -> int:
    """
    Apply saved rules to transactions where category = 'Other'.
    Returns number of rows updated.
    """
    rules = get_rules()
    if rules.empty:
        return 0

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    total_updated = 0

    for _, row in rules.iterrows():
        pattern = f"%{row['pattern']}%"
        cat = row["category"]

        cur.execute(
            """
            UPDATE transactions
            SET category = ?
            WHERE category = 'Other'
              AND description IS NOT NULL
              AND UPPER(description) LIKE ?
            """,
            (cat, pattern),
        )
        total_updated += cur.rowcount

    conn.commit()
    conn.close()
    return total_updated


def remove_duplicate_imported_csv() -> int:
    """
    Remove exact-duplicate rows for account='Imported CSV',
    keeping the first occurrence of each unique transaction.
    Returns the number of rows deleted.
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT id, tx_date, description, category, amount, tx_type, account
        FROM transactions
        WHERE account = 'Imported CSV'
        """,
        conn,
    )

    if df.empty:
        conn.close()
        return 0

    # Mark duplicates based on all key fields, keep first occurrence
    dup_mask = df.duplicated(
        subset=["tx_date", "description", "category", "amount", "tx_type", "account"],
        keep="first",
    )
    dup_ids = df.loc[dup_mask, "id"].tolist()

    if not dup_ids:
        conn.close()
        return 0

    cur = conn.cursor()
    placeholders = ",".join("?" * len(dup_ids))
    cur.execute(f"DELETE FROM transactions WHERE id IN ({placeholders})", dup_ids)
    conn.commit()
    conn.close()
    return len(dup_ids)


# =================== DATE HELPERS ===================

def get_month_bounds(dt: date):
    first = dt.replace(day=1)
    last_day = monthrange(dt.year, dt.month)[1]
    last = dt.replace(day=last_day)
    return first, last

# =============== SAVINGS + EXTRA DEBT HELPERS ===============

def get_period_adjustments():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM period_adjustments WHERE id = 1")
    row = cur.fetchone()
    conn.close()
    return dict(row)

def update_period_adjustments(savings: float, extra_debt: float):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "UPDATE period_adjustments SET savings = ?, extra_debt = ? WHERE id = 1",
        (savings, extra_debt),
    )
    conn.commit()
    conn.close()


# =================== BILLS & FIXED EXPENSES ===================

def get_bills_df():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM bills ORDER BY name", conn)
    conn.close()
    return df

def add_bill(name: str, amount: float, frequency: str = "monthly"):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO bills (name, amount, frequency) VALUES (?, ?, ?)",
        (name, amount, frequency),
    )
    conn.commit()
    conn.close()

def delete_bill(bill_id: int):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM bills WHERE id = ?", (bill_id,))
    conn.commit()
    conn.close()

def compute_monthly_bills():
    """For now we treat everything as monthly."""
    df = get_bills_df()
    if df.empty:
        return 0.0
    return float(df["amount"].sum())

def compute_per_paycheck_bills():
    """Spread monthly bills across your pay periods."""
    monthly_total = compute_monthly_bills()
    if monthly_total == 0:
        return 0.0
    settings = get_pay_period_settings()
    period_days = settings["period_days"]
    # Approx # of pay periods per year
    periods_per_year = 365 / period_days
    yearly_bills = monthly_total * 12
    return yearly_bills / periods_per_year



def get_pay_period_settings():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM pay_period_settings WHERE id = 1", conn)
    conn.close()
    row = df.iloc[0]
    return {
        "reference_payday": datetime.fromisoformat(row["reference_payday"]).date(),
        "period_days": int(row["period_days"]),
        "pot_budget": float(row["pot_budget"]),
    }

def save_pay_period_settings(settings):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE pay_period_settings
        SET reference_payday = ?, period_days = ?, pot_budget = ?
        WHERE id = 1
        """,
        (
            settings["reference_payday"].isoformat(),
            settings["period_days"],
            settings["pot_budget"],
        ),
    )
    conn.commit()
    conn.close()

def get_current_pay_period(today: date, settings=None):
    if settings is None:
        settings = get_pay_period_settings()
    ref = settings["reference_payday"]
    period_days = settings["period_days"]
    delta_days = (today - ref).days
    # handle dates before the reference sensibly
    cycle_index = delta_days // period_days if delta_days >= 0 else -((-delta_days - 1) // period_days) - 1
    start = ref + timedelta(days=cycle_index * period_days)
    end = start + timedelta(days=period_days - 1)
    return start, end

# =================== GAMIFICATION ===================

def get_gamification():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM gamification WHERE id = 1", conn)
    conn.close()
    row = df.iloc[0]
    return {
        "total_xp": int(row["total_xp"]),
        "streak": int(row["streak"]),
        "best_streak": int(row["best_streak"]),
        "last_activity_date": (
            datetime.fromisoformat(row["last_activity_date"]).date()
            if row["last_activity_date"]
            else None
        ),
    }

def save_gamification(stats):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE gamification
        SET total_xp = ?, streak = ?, best_streak = ?, last_activity_date = ?
        WHERE id = 1
        """,
        (
            stats["total_xp"],
            stats["streak"],
            stats["best_streak"],
            stats["last_activity_date"].isoformat()
            if stats["last_activity_date"]
            else None,
        ),
    )
    conn.commit()
    conn.close()

def xp_for_activity(activity_type: str) -> int:
    if activity_type == "transaction":
        return 5
    if activity_type == "under_pot":
        return 20
    if activity_type == "no_spend_day":
        return 25
    if activity_type == "savings":
        return 15
    if activity_type == "debt_payment":
        return 25
    return 0

def award_badge(name: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM badges WHERE name = ?", (name,))
    exists = cur.fetchone()[0] > 0
    if not exists:
        cur.execute(
            "INSERT INTO badges (name, earned_date) VALUES (?, ?)",
            (name, date.today().isoformat()),
        )
        conn.commit()
    conn.close()
    return not exists

def get_badges():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT name, earned_date FROM badges ORDER BY earned_date", conn)
    conn.close()
    return df

def update_gamification_on_activity():
    today = date.today()
    stats = get_gamification()
    last = stats["last_activity_date"]

    # XP for logging a transaction
    stats["total_xp"] += xp_for_activity("transaction")

    # Streak logic
    if last is None:
        stats["streak"] = 1
    else:
        if today == last:
            # same day, nothing changes
            pass
        elif today == last + timedelta(days=1):
            stats["streak"] += 1
        elif today > last:
            stats["streak"] = 1

    if stats["streak"] > stats["best_streak"]:
        stats["best_streak"] = stats["streak"]

    stats["last_activity_date"] = today
    save_gamification(stats)

    # Badge: First 10 transactions
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM transactions")
    tx_count = cur.fetchone()[0]
    conn.close()
    if tx_count >= 10:
        award_badge("First 10 Transactions")

    # Badge: 3-day streak
    if stats["streak"] >= 3:
        award_badge("3-Day Streak")

def compute_level(total_xp: int) -> int:
    # Simple: 100 XP per level
    return max(1, total_xp // 100 + 1)

# =================== SUMMARY & IMPORT ===================

def compute_summary(df: pd.DataFrame):
    if df.empty:
        return {
            "income": 0.0,
            "expenses": 0.0,
            "net": 0.0,
            "by_category": pd.DataFrame(),
        }

    income = df.loc[df["tx_type"] == "income", "amount"].sum()
    expenses = df.loc[df["tx_type"] == "expense", "amount"].sum()
    net = income - expenses

    exp_df = df[df["tx_type"] == "expense"]
    if not exp_df.empty:
        cat = (
            exp_df.groupby("category")["amount"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )
    else:
        cat = pd.DataFrame(columns=["category", "amount"])

    return {
        "income": income,
        "expenses": expenses,
        "net": net,
        "by_category": cat,
    }

def import_csv_to_transactions(uploaded_file, default_type="expense"):
    df = pd.read_csv(uploaded_file)

    col_map = {"date": None, "description": None, "amount": None}

    for col in df.columns:
        low = col.lower()
        if "date" in low and col_map["date"] is None:
            col_map["date"] = col
        if ("descr" in low or "memo" in low or "detail" in low) and col_map["description"] is None:
            col_map["description"] = col
        if ("amount" in low or "amt" in low) and col_map["amount"] is None:
            col_map["amount"] = col

    missing = [k for k, v in col_map.items() if v is None]
    if missing:
        raise ValueError(f"Could not auto-detect columns for: {missing}. Please adjust code or clean CSV.")

    for _, row in df.iterrows():
        try:
            tx_date = pd.to_datetime(row[col_map["date"]]).date().isoformat()
        except Exception:
            continue

        description = str(row[col_map["description"]])
        amount = float(row[col_map["amount"]])

        if amount < 0:
            tx_type = "expense"
            amount = abs(amount)
        else:
            tx_type = default_type

        category = "Other"
        add_transaction(tx_date, description, category, amount, tx_type, account="Imported CSV")

# =================== PAY PERIOD SUMMARY ===================

def get_pay_period_summary(today: date):
    settings = get_pay_period_settings()
    start, end = get_current_pay_period(today, settings)
    df_period = load_transactions(start, end)
    if df_period.empty:
        spent = 0.0
        spent_weekly_cats = 0.0
    else:
        spent = df_period.loc[df_period["tx_type"] == "expense", "amount"].sum()
        in_weekly = df_period[
            (df_period["tx_type"] == "expense")
            & (df_period["category"].isin(WEEKLY_CATEGORIES))
        ]
        spent_weekly_cats = in_weekly["amount"].sum()

    pot_budget = settings["pot_budget"]
    leftover = pot_budget - spent_weekly_cats
    days_total = settings["period_days"]
    days_passed = (min(today, end) - start).days + 1
    days_left = max(0, days_total - days_passed)
    return {
        "start": start,
        "end": end,
        "spent_total": spent,
        "spent_weekly": spent_weekly_cats,
        "pot_budget": pot_budget,
        "leftover": leftover,
        "days_total": days_total,
        "days_passed": days_passed,
        "days_left": days_left,
    }

# =================== PET LEVELS ===================

def pet_levels():
    stats = get_gamification()
    total_xp = stats["total_xp"]
    level_player = compute_level(total_xp)

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT category, tx_type, amount FROM transactions", conn
    )
    conn.close()

    if df.empty:
        bc_level = 1
        cs_level = 1
    else:
        disciplined_exp = df[
            (df["tx_type"] == "expense") & (df["category"].isin(WEEKLY_CATEGORIES))
        ]["amount"].sum()
        savings_total = df[
            (df["tx_type"] == "expense") & (df["category"] == "Savings")
        ]["amount"].sum()
        # scale these into 1–20 ranges with a soft curve
        bc_level = max(1, min(20, int(disciplined_exp ** 0.25)))
        cs_level = max(1, min(20, int((abs(savings_total) + total_xp / 10) ** 0.25)))

    def border_collie_ascii(lv: int) -> str:
        if lv < 5:
            return " /ᐠ｡ꞈ｡ᐟ\\  Border Collie Lv." + str(lv)
        elif lv < 10:
            return " /ᐠ•▾•ᐟ\\  🥏  Border Collie Lv." + str(lv)
        elif lv < 15:
            return " /ᐠ•ᴥ•ᐟ\\  🥇  Border Collie Lv." + str(lv)
        else:
            return " /ᐠ🔥▾🔥ᐟ\\  🐺  Border Collie Lv." + str(lv)

    def cocker_spaniel_ascii(lv: int) -> str:
        if lv < 5:
            return " (ᵔᴥᵔ)  Cocker Spaniel Lv." + str(lv)
        elif lv < 10:
            return " (ᵔ◡ᵔ)っ💰  Cocker Spaniel Lv." + str(lv)
        elif lv < 15:
            return " (✿◠‿◠)💎  Cocker Spaniel Lv." + str(lv)
        else:
            return " (｡◕‿◕｡)✨🏰  Cocker Spaniel Lv." + str(lv)

    return {
        "player_level": level_player,
        "border_collie_level": bc_level,
        "border_collie_art": border_collie_ascii(bc_level),
        "cocker_spaniel_level": cs_level,
        "cocker_spaniel_art": cocker_spaniel_ascii(cs_level),
    }

# =================== MONEY COACH ===================

def money_coach_suggestions(today: date):
    tips = []
    pp = get_pay_period_summary(today)
    pot = pp["pot_budget"]
    spent = pp["spent_weekly"]
    leftover = pp["leftover"]
    days_left = pp["days_left"]

    if pot > 0:
        if leftover < 0:
            tips.append("You're over your pot for this pay period. Try a no-spend day to earn bonus XP and let your budget catch up.")
        else:
            per_day_safe = leftover / max(1, days_left) if days_left > 0 else leftover
            tips.append(
                f"You have ${leftover:,.2f} left in this pot. That's about ${per_day_safe:,.2f} per day for the rest of the period."
            )

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT tx_date, category, amount, tx_type FROM transactions", conn
    )
    conn.close()
    if not df.empty:
        df["tx_date"] = pd.to_datetime(df["tx_date"]).dt.date
        recent = df[df["tx_date"] >= today - timedelta(days=14)]

        eat_out = recent[
            (recent["tx_type"] == "expense")
            & (recent["category"] == "Eating Out")
        ]["amount"].sum()
        groceries = recent[
            (recent["tx_type"] == "expense")
            & (recent["category"] == "Groceries")
        ]["amount"].sum()
        if eat_out > 0 and groceries > 0 and eat_out > 0.5 * groceries:
            tips.append(
                "Eating Out is more than half of your Groceries in the last 2 weeks. Try swapping 1–2 meals to homemade for extra savings + XP."
            )

        debt = recent[
            (recent["tx_type"] == "expense")
            & (recent["category"] == "Debt Payoff (Jenius Loan)")
        ]["amount"].sum()
        if debt > 0:
            tips.append(
                f"You've paid ${debt:,.2f} toward your Jenius loan recently. Consider rounding up the next payment for extra progress (+XP & a future badge)."
            )

        savings = recent[
            (recent["tx_type"] == "expense")
            & (recent["category"] == "Savings")
        ]["amount"].sum()
        if savings > 0:
            tips.append(
                f"You've moved ${savings:,.2f} into Savings recently. Keeping this streak going will level up your Cocker Spaniel faster."
            )

    if not tips:
        tips.append("Log a few days of transactions and I'll start giving you tailored money tips here.")

    return tips

def import_transactions_from_df(
    df,
    date_col: str,
    desc_col: str,
    mode: str,
    amount_col: str | None,
    debit_col: str | None,
    credit_col: str | None,
    default_type: str = "expense",
):
    """
    Import transactions from a DataFrame using user-selected columns.

    mode:
      - "single": use amount_col (+/-)
      - "split": use separate debit_col / credit_col
    """
    for _, row in df.iterrows():
        # Parse date
        try:
            dt = pd.to_datetime(row[date_col], errors="coerce")
        except Exception:
            continue
        if pd.isna(dt):
            continue
        tx_date = dt.date().isoformat()

        description = str(row[desc_col]) if desc_col is not None else ""

        if mode == "single":
            if amount_col is None:
                continue
            try:
                amt_raw = float(row[amount_col])
            except Exception:
                continue
            if amt_raw == 0 or pd.isna(amt_raw):
                continue

            if amt_raw < 0:
                tx_type = "expense"
                amount = abs(amt_raw)
            else:
                tx_type = default_type  # "expense" or "income" depending on your radio
                amount = amt_raw

            category = "Other"
            add_transaction(tx_date, description, category, amount, tx_type, account="Imported CSV")

        elif mode == "split":
            # Debit and credit separate
            debit_val = 0.0
            credit_val = 0.0

            if debit_col is not None:
                try:
                    v = row[debit_col]
                    debit_val = float(v) if not pd.isna(v) else 0.0
                except Exception:
                    debit_val = 0.0

            if credit_col is not None:
                try:
                    v = row[credit_col]
                    credit_val = float(v) if not pd.isna(v) else 0.0
                except Exception:
                    credit_val = 0.0

            # Debit = money going out (expense)
            if debit_val > 0:
                category = "Other"
                add_transaction(
                    tx_date,
                    description,
                    category,
                    debit_val,
                    "expense",
                    account="Imported CSV",
                )

            # Credit = money coming in (income / refund)
            if credit_val > 0:
                tx_type = "income" if default_type == "income" else default_type
                category = "Other"
                add_transaction(
                    tx_date,
                    description,
                    category,
                    credit_val,
                    tx_type,
                    account="Imported CSV",
                )


# =================== MAIN APP ===================

def main():
    st.set_page_config(
        page_title="Budget Commander (Gamified)",
        page_icon="🕹️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_db()

    today = date.today()
    st.title("🕹️ Budget Commander v0.3 — Paycheck Pot Edition")
    st.caption("Biweekly pot tracking + XP, pets, and money coaching for you and your wife.")

    stats = get_gamification()
    level = compute_level(stats["total_xp"])
    pets = pet_levels()

    # ---------- SIDEBAR: PROFILE ----------
    st.sidebar.header("🎮 Game Profile")
    st.sidebar.metric("Player Level", level)
    st.sidebar.metric("XP", stats["total_xp"])
    st.sidebar.metric("🔥 Streak (days)", stats["streak"])
    st.sidebar.metric("🏆 Best Streak", stats["best_streak"])

    st.sidebar.markdown("---")
    st.sidebar.markdown("**🐶 Your Money Dogs**")
    st.sidebar.text(pets["border_collie_art"])
    st.sidebar.text(pets["cocker_spaniel_art"])

    # ---------- SIDEBAR: PAY PERIOD POT ----------
    st.sidebar.markdown("---")
    st.sidebar.header("💰 Pay Period Pot Settings")
    settings = get_pay_period_settings()
    with st.sidebar.form("pot_settings_form"):
        ref_payday = st.date_input(
            "Reference payday (start of cycle)", value=settings["reference_payday"]
        )
        period_days = st.number_input(
            "Days per cycle", min_value=7, max_value=31, value=settings["period_days"]
        )
        pot_budget = st.number_input(
            "Discretionary pot per cycle ($)",
            min_value=0.0,
            step=10.0,
            value=float(settings["pot_budget"]),
            format="%.2f",
        )
        save_pot = st.form_submit_button("Save pot settings")
        if save_pot:
            settings["reference_payday"] = ref_payday
            settings["period_days"] = int(period_days)
            settings["pot_budget"] = float(pot_budget)
            save_pay_period_settings(settings)
            st.sidebar.success("Pay period settings updated!")

    # ---------- SIDEBAR: ADD TRANSACTION ----------
    st.sidebar.markdown("---")
    st.sidebar.header("➕ Add Transaction")
    with st.sidebar.form("add_tx_form", clear_on_submit=True):
        tx_date = st.date_input("Date", value=today)
        description = st.text_input("Description", "")
        category = st.selectbox(
            "Category",
            WEEKLY_CATEGORIES + MONTHLY_CATEGORIES + ["Other"],
        )
        tx_type = st.radio("Type", ["Expense", "Income"], horizontal=True)
        amount = st.number_input("Amount", min_value=0.0, step=1.0, format="%.2f")
        account = st.text_input("Account (optional)", "")

        submitted = st.form_submit_button("Save")
        if submitted:
            if amount <= 0:
                st.sidebar.error("Amount must be greater than 0.")
            else:
                norm_type = "expense" if tx_type.lower() == "expense" else "income"
                add_transaction(
                    tx_date.isoformat(),
                    description,
                    category,
                    amount,
                    norm_type,
                    account,
                )
                st.sidebar.success("Transaction saved! +5 XP")

    # ---------- TABS ----------
    tab_dash, tab_tx, tab_bills, tab_import, tab_wheel, tab_badges, tab_coach = st.tabs(
        ["📊 Dashboard", "📜 Transactions", "🏦 Bills & Fixed", "📂 Import CSV", "🎡 Spin Wheel", "🏅 Badges", "🧠 Money Coach"]
    )

    # ===== DASHBOARD TAB =====
    with tab_dash:
        st.subheader("📅 Time Filters")
        month_start, month_end = get_month_bounds(today)
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            mode = st.selectbox("View mode", ["This Month", "Today", "Custom Range"])

        if mode == "This Month":
            start_date, end_date = month_start, month_end
        elif mode == "Today":
            start_date = end_date = today
        else:
            with col_f2:
                start_date = st.date_input("Start date", value=month_start, key="start_custom")
            with col_f3:
                end_date = st.date_input("End date", value=month_end, key="end_custom")
            if start_date > end_date:
                st.warning("Start date is after end date. Swapping them.")
                start_date, end_date = end_date, start_date
        
        
        st.markdown("---")
        st.subheader("🍯 Paycheck Pot View")

        # Compute income and expenses for the *current* pay period
        pp_start, pp_end = get_current_pay_period(today)
        df_pp = load_transactions(start_date=pp_start, end_date=pp_end)

        if df_pp.empty:
            st.info("No transactions found for the current pay period yet.")
        else:
            income_total = df_pp[df_pp["tx_type"] == "income"]["amount"].sum()
            expense_total = df_pp[df_pp["tx_type"] == "expense"]["amount"].sum()
            if pot < 0:
                pot = 0.0  # don't show negative pot, just zero it

            remaining_pot = pot - expense_total
            remaining_pot = max(remaining_pot, 0.0)

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Income this pay period", f"${income_total:,.2f}")
            with col_b:
                st.metric("Bills share this period", f"${per_paycheck_bills:,.2f}")
            with col_c:
                st.metric("Spending pot (after bills)", f"${pot:,.2f}")

            st.metric("Remaining pot (variable spending left)", f"${remaining_pot:,.2f}")

            adj = get_period_adjustments()
            savings_amt = adj["savings"]
            extra_debt_amt = adj["extra_debt"]

            pot = income_total - per_paycheck_bills - savings_amt - extra_debt_amt


            days_left = (pp_end - today).days + 1
            if days_left > 0:
                safe_per_day = remaining_pot / days_left
                st.caption(f"You have about ${safe_per_day:,.2f} per day for the next {days_left} day(s).")

        st.markdown("### 🧭 Pay-Period Savings & Debt Adjustments")

        adj = get_period_adjustments()

        colA, colB = st.columns(2)
        with colA:
            savings_input = st.number_input(
                "Savings to set aside this pay period",
                min_value=0.0,
                value=float(adj["savings"]),
                step=10.0,
                format="%.2f",
            )
        with colB:
            debt_input = st.number_input(
                "Extra debt payoff this period",
                min_value=0.0,
                value=float(adj["extra_debt"]),
                step=10.0,
                format="%.2f",
            )

        if st.button("Save adjustments"):
            update_period_adjustments(savings_input, debt_input)
            st.success("Saved! These will now be included in your pot calculation.")



        df = load_transactions(start_date, end_date)
        summary = compute_summary(df)

        st.markdown("### 📊 Overview")
        c1, c2, c3 = st.columns(3)
        c1.metric("Income", f"${summary['income']:,.2f}")
        c2.metric("Expenses", f"${summary['expenses']:,.2f}")
        c3.metric("Net", f"${summary['net']:,.2f}")

        # Pay period pot
        st.markdown("### 💰 Current Pay Period Pot")
        pp = get_pay_period_summary(today)
        pot_cols = st.columns(4)
        pot_cols[0].metric("Pot Budget", f"${pp['pot_budget']:,.2f}")
        pot_cols[1].metric("Spent (weekly cats)", f"${pp['spent_weekly']:,.2f}")
        pot_cols[2].metric("Leftover", f"${pp['leftover']:,.2f}")
        pot_cols[3].metric("Days Left", pp["days_left"])

        if pp["pot_budget"] > 0:
            used_ratio = min(1.0, max(0.0, pp["spent_weekly"] / pp["pot_budget"]))
            st.write("**Spend Meter (pot usage this pay period)**")
            st.progress(used_ratio)
            if used_ratio < 0.5:
                st.success("You're cruising. Plenty of fuel left in this pot. 🟢")
            elif used_ratio < 0.9:
                st.warning("You're over halfway through this pot. Coast a bit to stay in the green. 🟡")
            else:
                if pp["leftover"] >= 0:
                    st.warning("Pot is almost used, but you're still technically under. Tight but winning. 🟠")
                else:
                    st.error("You've gone over this pot. Time for a no-spend challenge to recover. 🔴")

        st.caption(f"Current pay period: {pp['start'].isoformat()} → {pp['end'].isoformat()}")

        st.markdown("### 🧩 Spending by Category (Expenses Only)")
        if not summary["by_category"].empty:
            cat_df = summary["by_category"].set_index("category")
            st.bar_chart(cat_df)
        else:
            st.info("No expense data for this period yet. Add some transactions to see charts.")

        st.markdown("### 💡 Quick Insights")
        if df.empty:
            st.write("Once you have some data, I'll highlight patterns here.")
        else:
            total_days = (end_date - start_date).days + 1
            total_expenses = summary["expenses"]
            daily_burn = total_expenses / total_days if total_days > 0 else 0
            st.write(f"- Average **daily spending**: **${daily_burn:,.2f}**")
            if daily_burn > 0:
                monthly_proj = daily_burn * 30
                st.write(
                    f"- If you keep this pace, projected **monthly spending** is about **${monthly_proj:,.0f}**."
                )
            if not summary["by_category"].empty:
                top_cat = summary["by_category"].iloc[0]
                st.write(
                    f"- Top category: **{top_cat['category']}** — **${top_cat['amount']:,.2f}**."
                )
            if summary["net"] < 0:
                st.write("🔴 You are spending more than your income in this period.")
            elif summary["net"] > 0:
                st.write("🟢 You’re net positive this period. Great job building XP IRL.")

        # ===== TRANSACTIONS TAB =====
    with tab_tx:
        st.subheader("📜 All Transactions")
        df_all = load_transactions()
        if df_all.empty:
            st.write("No transactions logged yet.")
        else:
            st.write("Tip: Use the cleaner below to re-categorize imported 'Other' items.")

            # Pretty table for viewing
            show_df = df_all.copy()
            show_df["amount"] = show_df["amount"].map(lambda x: f"${x:,.2f}")
            show_df = show_df.rename(
                columns={
                    "tx_date": "Date",
                    "description": "Description",
                    "category": "Category",
                    "amount": "Amount",
                    "tx_type": "Type",
                    "account": "Account",
                }
            )
            st.dataframe(show_df, use_container_width=True, hide_index=True)
        # ===== BILLS & FIXED TAB =====
    with tab_bills:
        st.subheader("🏦 Bills & Fixed Monthly Expenses")

        st.markdown("These are the things that come off the top: mortgage, car, insurance, etc.")

        # Add new bill
        with st.form("add_bill_form", clear_on_submit=True):
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                name = st.text_input("Bill name", placeholder="Mortgage, Car Payment, Internet...")
            with col2:
                amount = st.number_input("Monthly amount", min_value=0.0, step=10.0, format="%.2f")
            with col3:
                frequency = st.selectbox("Frequency", ["monthly"], index=0)

            submitted = st.form_submit_button("Add / Save Bill")
            if submitted:
                if not name or amount <= 0:
                    st.warning("Please enter a name and amount.")
                else:
                    add_bill(name, amount, frequency)
                    st.success(f"Saved bill: {name} (${amount:.2f}/month)")

        # Show existing bills
        bills_df = get_bills_df()
        if bills_df.empty:
            st.info("No bills added yet. Start by adding mortgage, car, insurance, etc.")
        else:
            st.markdown("### Current Bills")
            show_df = bills_df.copy()
            show_df = show_df.rename(columns={"name": "Bill", "amount": "Monthly Amount", "frequency": "Frequency"})
            show_df["Monthly Amount"] = show_df["Monthly Amount"].map(lambda x: f"${x:,.2f}")
            st.dataframe(show_df[["Bill", "Monthly Amount", "Frequency"]], use_container_width=True, hide_index=True)

            monthly_total = compute_monthly_bills()
            per_paycheck = compute_per_paycheck_bills()
            st.markdown(
                f"**Total monthly bills:** ${monthly_total:,.2f}  \n"
                f"**Bills per pay period (approx):** ${per_paycheck:,.2f}"
            )

            # Simple delete control
            bill_ids = bills_df["id"].tolist()
            bill_labels = [f"{row['name']} (${row['amount']:.2f})" for _, row in bills_df.iterrows()]
            del_choice = st.selectbox("Delete a bill", ["-- Select --"] + bill_labels)
            if del_choice != "-- Select --":
                idx = bill_labels.index(del_choice)
                del_id = bill_ids[idx]
                if st.button("Confirm delete"):
                    delete_bill(del_id)
                    st.success("Bill deleted. Reload the page to see updated list.")


                       # ---------- CLEAN & CATEGORIZE ----------
            with st.expander("🧹 Clean & Categorize Transactions"):
                st.markdown("Refine and correct your transactions.")

                # Filters
                df_clean = load_transactions()  # ensure df_clean exists
                search = st.text_input("Search description")
                existing_cats = sorted(df_clean["category"].dropna().astype(str).unique().tolist())
                default_filter = ["Other"] if "Other" in existing_cats else []

                cat_filter = st.multiselect(
                    "Filter by current category",
                    options=existing_cats,
                    default=default_filter,
                )

                # Filter logic
                work_df = df_clean.copy()
                if search:
                    work_df = work_df[
                        work_df["description"].astype(str).str.contains(search, case=False)
                    ]
                if cat_filter:
                    work_df = work_df[work_df["category"].isin(cat_filter)]

                # No results
                if work_df.empty:
                    st.info("No transactions match your filters.")
                else:
                    st.write(f"{len(work_df)} transaction(s) match your filters.")

                    # Build labels
                    option_labels = []
                    label_to_id = {}
                    for _, row in work_df.iterrows():
                        label = (
                            f"#{row['id']} | {row['tx_date']} | {row['category']} | "
                            f"${row['amount']:.2f} | {str(row['description'])[:40]}"
                        )
                        option_labels.append(label)
                        label_to_id[label] = int(row["id"])

                    selected_labels = st.multiselect(
                        "Select transactions to recategorize",
                        options=option_labels,
                    )
                    selected_ids = [label_to_id[l] for l in selected_labels]

                    # Category choice
                    new_cat = st.selectbox(
                        "New category",
                        WEEKLY_CATEGORIES + MONTHLY_CATEGORIES + ["Other"],
                    )

                    # Suggest rule
                    suggested_pattern = ""
                    if selected_ids:
                        first_id = selected_ids[0]
                        first_desc = (
                            work_df.loc[work_df["id"] == first_id, "description"]
                            .astype(str)
                            .iloc[0]
                        )
                        suggested_pattern = first_desc.split()[0][:20].upper()

                    rule_pattern = st.text_input(
                        "Optional: Save a rule (text to match in description)",
                        value=suggested_pattern,
                        help="Example: WALMART, SHELL, MCDONALD — future matching 'Other' transactions auto-categorize.",
                    )

                    # Apply
                    if st.button("Apply new category to selected"):
                        if not selected_ids:
                            st.warning("Select at least one transaction.")
                        else:
                            update_transaction_categories(selected_ids, new_cat)

                            msg = f"Updated {len(selected_ids)} transaction(s) to category '{new_cat}'."
                            if rule_pattern.strip():
                                add_or_update_rule(rule_pattern.strip().upper(), new_cat)
                                msg += f" Saved rule: '{rule_pattern.strip().upper()}' → {new_cat}."
                            st.success(msg)
                            st.rerun()

                st.markdown("---")

                # Auto categorize using saved rules
                if st.button("⚡ Auto-categorize 'Other' using saved rules"):
                    updated = apply_rules_to_other()
                    if updated > 0:
                        st.success(f"Auto-categorized {updated} 'Other' transaction(s).")
                        st.rerun()
                    else:
                        st.info("No 'Other' transactions matched any rules.")




    # ===== IMPORT CSV TAB =====
    with tab_import:
        st.subheader("📂 Import Bank/Credit Card CSV")
        st.write(
            "Download a CSV from your bank/credit card website and upload it here.\n"
            "Then choose which columns are Date, Description, and Amount/Debit/Credit."
        )

        uploaded = st.file_uploader("Upload CSV", type=["csv"])

        if uploaded is not None:
            # Read once and reuse
            df_raw = pd.read_csv(uploaded)
            st.write("Preview of your file (first 10 rows):")
            st.dataframe(df_raw.head(10), use_container_width=True)

            if df_raw.empty:
                st.warning("This file appears to be empty.")
            else:
                cols = list(df_raw.columns)

                st.markdown("### 🧩 Column Mapping")

                date_col = st.selectbox("Date column", cols)
                desc_col = st.selectbox("Description column", cols)

                amount_mode = st.radio(
                    "How are amounts stored?",
                    [
                        "Single amount column (+/- for credit/debit)",
                        "Separate Debit and Credit columns",
                    ],
                    horizontal=False,
                )

                amount_col = None
                debit_col = None
                credit_col = None

                if amount_mode.startswith("Single"):
                    amount_col = st.selectbox("Amount column", cols)
                else:
                    debit_col = st.selectbox("Debit column (money you spent)", cols)
                    credit_col = st.selectbox("Credit column (money you received/refunds)", cols)

                default_type_choice = st.radio(
                    "Default type for positive amounts (when sign isn't clear)",
                    ["Expense", "Income"],
                    horizontal=True,
                )
                default_type = "expense" if default_type_choice == "Expense" else "income"

                if st.button("Import CSV"):
                    try:
                        mode = "single" if amount_mode.startswith("Single") else "split"
                        import_transactions_from_df(
                            df_raw,
                            date_col=date_col,
                            desc_col=desc_col,
                            mode=mode,
                            amount_col=amount_col,
                            debit_col=debit_col,
                            credit_col=credit_col,
                            default_type=default_type,
                        )
                        st.success("Imported CSV and logged transactions. XP awarded for activity! 🎉")
                    except Exception as e:
                        st.error(f"Import failed: {e}")
        else:
            st.info("Upload a CSV file to begin mapping columns.")

        st.markdown("---")
        st.subheader("🗑️ Clean up duplicate imported rows")
        st.write(
            "If you accidentally imported the same CSV twice, use this to remove exact duplicates "
            "for account = 'Imported CSV' (keeps one copy of each)."
        )
        if st.button("Remove duplicate Imported CSV rows"):
            removed = remove_duplicate_imported_csv()
            if removed > 0:
                st.success(f"Removed {removed} duplicate imported row(s).")
                st.rerun()
            else:
                st.info("No exact duplicate imported rows found to remove.")


    # ===== SPIN WHEEL TAB =====
    with tab_wheel:
        st.subheader("🎡 Spin-the-Wheel Category Selector")
        st.write("Use this when you're about to spend and want to make it a little game.")
        st.write("Wheel is weighted toward your weekly categories.")
        if st.button("🎡 Spin the Wheel!"):
            choices = WEEKLY_CATEGORIES * 3 + MONTHLY_CATEGORIES
            pick = random.choice(choices)
            st.markdown(f"## 🎯 The wheel landed on: **{pick}**")
            if pick in WEEKLY_CATEGORIES:
                st.write("This fits right into your pay-period pot. Log it and earn XP.")
            else:
                st.write("This is more of a monthly-category expense. Keep an eye on big-picture trends.")

    # ===== BADGES TAB =====
    with tab_badges:
        st.subheader("🏅 Badges")
        df_badges = get_badges()
        if df_badges.empty:
            st.write("No badges earned yet. Start logging, saving, and staying under budget to earn some!")
        else:
            for _, row in df_badges.iterrows():
                st.markdown(f"- **{row['name']}** — earned on {row['earned_date']}")

    # ===== MONEY COACH TAB =====
    with tab_coach:
        st.subheader("🧠 Money Coach")
        tips = money_coach_suggestions(today)
        for tip in tips:
            st.markdown(f"- {tip}")


if __name__ == "__main__":
    main()


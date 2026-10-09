import pandas as pd
from rapidfuzz import fuzz
from itertools import combinations


def run_audit(beneficiaries, bank_accounts, disbursements):
    # Validate required columns
    required = {
        "beneficiaries": [
            "beneficiary_id", "beneficiary_name", "date_of_birth"
        ],
        "bank_accounts": [
            "beneficiary_id", "account_id"
        ],
        "disbursements": [
            "transaction_id", "beneficiary_id", "account_id", "amount"
        ],
    }

    datasets = {
        "beneficiaries": beneficiaries,
        "bank_accounts": bank_accounts,
        "disbursements": disbursements,
    }

    for dataset_name, columns in required.items():
        missing = [
            column for column in columns
            if column not in datasets[dataset_name].columns
        ]
        if missing:
            raise ValueError(
                f"{dataset_name} is missing columns: {missing}"
            )

    # 1. Shared payout accounts
    shared_accounts = (
        bank_accounts.groupby("account_id")
        .agg(
            beneficiary_count=("beneficiary_id", "nunique"),
            beneficiaries=("beneficiary_id", list),
        )
        .reset_index()
    )

    shared_accounts = shared_accounts[
        shared_accounts["beneficiary_count"] > 1
    ]

    # 2. Potential identity matches
    identity_matches = []

    identity_df = beneficiaries[
        ["beneficiary_id", "beneficiary_name", "date_of_birth"]
    ].dropna(subset=["beneficiary_id", "beneficiary_name"]).copy()

    identity_df["beneficiary_name"] = (
        identity_df["beneficiary_name"].astype(str).str.strip()
    )
    identity_df["date_of_birth"] = (
        identity_df["date_of_birth"].fillna("").astype(str).str.strip()
    )

    records = identity_df.to_dict("records")

    for a, b in combinations(records, 2):
        if a["beneficiary_id"] == b["beneficiary_id"]:
            continue

        score = fuzz.ratio(
            a["beneficiary_name"].lower(),
            b["beneficiary_name"].lower(),
        )

        same_dob = (
            bool(a["date_of_birth"])
            and a["date_of_birth"] == b["date_of_birth"]
        )

        if score >= 85 or same_dob:
            identity_matches.append({
                "beneficiary_1": str(a["beneficiary_id"]),
                "name_1": a["beneficiary_name"],
                "beneficiary_2": str(b["beneficiary_id"]),
                "name_2": b["beneficiary_name"],
                "name_similarity": score,
                "same_dob": same_dob,
            })

    # 3. Payment summaries
    payments = disbursements.copy()
    payments["amount"] = pd.to_numeric(
        payments["amount"], errors="coerce"
    )

    shared_ids = set(shared_accounts["account_id"].astype(str))
    payments["shared_account_flag"] = (
        payments["account_id"].astype(str).isin(shared_ids)
    )

    payment_summary = (
        payments.groupby("beneficiary_id")
        .agg(
            transaction_count=("transaction_id", "nunique"),
            total_amount=("amount", "sum"),
            average_amount=("amount", "mean"),
            shared_account_payments=("shared_account_flag", "sum"),
        )
        .reset_index()
    )

    # 4. Combine identity and account evidence into review pairs
    account_pairs = []

    for _, row in shared_accounts.iterrows():
        ids = list(dict.fromkeys(map(str, row["beneficiaries"])))

        for a, b in combinations(sorted(ids), 2):
            account_pairs.append({
                "beneficiary_1": a,
                "beneficiary_2": b,
                "shared_account": str(row["account_id"]),
            })

    identity_df_out = pd.DataFrame(identity_matches)
    account_pairs_df = pd.DataFrame(
        account_pairs,
        columns=["beneficiary_1", "beneficiary_2", "shared_account"],
    )

    # Normalize pair ordering before merging
    def normalize_pairs(df):
        df = df.copy()
        if df.empty:
            df["pair_key"] = pd.Series(dtype=object)
            return df

        df["pair_key"] = df.apply(
            lambda r: tuple(sorted([
                str(r["beneficiary_1"]),
                str(r["beneficiary_2"]),
            ])),
            axis=1,
        )
        return df

    identity_df_out = normalize_pairs(identity_df_out)
    account_pairs_df = normalize_pairs(account_pairs_df)

    review_pairs = pd.merge(
        identity_df_out,
        account_pairs_df[["pair_key", "shared_account"]],
        on="pair_key",
        how="outer",
    )

    # Avoid assigning an identity score to account-only pairs
    review_pairs["risk_score"] = (
        review_pairs["name_similarity"].ge(85).fillna(False).astype(int) * 30
        + review_pairs["same_dob"].fillna(False).astype(int) * 30
        + review_pairs["shared_account"].notna().astype(int) * 40
    )

    review_pairs["risk_level"] = pd.cut(
        review_pairs["risk_score"],
        bins=[-1, 29, 59, 100],
        labels=["Low", "Medium", "High"],
    ).astype(str)

    # Convert NaN values to JSON-safe None
    def records_for_json(df):
        return df.astype(object).where(pd.notna(df), None).to_dict("records")

    return {
        "summary": {
            "beneficiaries": int(beneficiaries["beneficiary_id"].nunique()),
            "shared_accounts": int(len(shared_accounts)),
            "potential_duplicate_pairs": int(len(identity_matches)),
            "transactions": int(len(disbursements)),
            "review_pairs": int(len(review_pairs)),
        },
        "shared_accounts": records_for_json(shared_accounts),
        "identity_matches": records_for_json(identity_df_out.drop(columns=["pair_key"])),
        "payment_summary": records_for_json(payment_summary),
        "review_pairs": records_for_json(review_pairs.drop(columns=["pair_key"])),
    }
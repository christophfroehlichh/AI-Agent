from datetime import datetime
from tools.checks import compare_time_periods_with_llm

def run_manual_tests():
    cases = [
    # ==========================================
    # 1) Gleich (4 Tage erwartet)
    # ==========================================
    (
        "Time Period: 2024-05-02 – 2024-05-05",
        "Time Period 2024-05-02 – 2024-05-05",
        "gleich (4 Tage erwartet)",
    ),

    # ==========================================
    # 2) Summary länger (6 Tage erwartet)
    # ==========================================
    (
        "Time Period: 2024-05-02 – 2024-05-05",
        "Time Period 2024-05-02 – 2024-05-07",
        "Summary länger (6 Tage erwartet)",
    ),

    # ==========================================
    # 3) Summary länger (4 Tage erwartet)
    # ==========================================
    (
        "Time Period: 2025-07-08 – 2025-07-09",
        "Time Period 2025-07-08 – 2025-07-11",
        "Summary länger (4 Tage erwartet)",
    ),

    # ==========================================
    # 4) Header länger (7 Tage erwartet)
    # ==========================================
    (
        "Time Period: 2024-06-01 – 2024-06-07",
        "Time Period: 2024-06-03 – 2024-06-05",
        "Header länger (7 Tage erwartet)",
    ),

    # ==========================================
    # 7) Verschiedene Bindestriche / Typo / Whitespace
    # ==========================================
    (
        "TimePeriod 2024-05-01 — 2024-05-03",      # em dash
        "Time Period : 2024-05-01-2024-05-04",     # missing separators + typo
        "Summary länger (4 Tage erwartet)",
    ),

    # ==========================================
    # 8) Beide gültig aber Schreibweise komplett anders
    # ==========================================
    (
        "TP: 2024-09-10   –     2024-09-14",
        "timeperiod=2024-09-10--2024-09-12",
        "Header länger (5 Tage erwartet)",
    ),

    # ==========================================
    # 9) Header hat vertauschte Daten → Python fixen!
    # ==========================================
    (
        "Time Period: 2024-12-20 – 2024-12-10",  # End < Start
        "Time Period: 2024-12-10 – 2024-12-12",
        "Header nach Korrektur länger (11 Tage erwartet)",
    ),

    # ==========================================
    # 10) Summary hat vertauschte Daten
    # ==========================================
    (
        "Time Period: 2024-12-01 – 2024-12-03",
        "Time Period: 2024-12-05 – 2024-11-30",
        "Summary nach Korrektur länger (6 Tage erwartet)",
    ),

    # ==========================================
    # 12) Header ohne Präfix
    # ==========================================
    (
        "2024-03-01 – 2024-03-05",
        "Period: 2024-03-01 – 2024-03-03",
        "Header länger (5 Tage erwartet)",
    ),

    # ==========================================
    # 13) Summary ohne Präfix
    # ==========================================
    (
        "Time Period: 2024-02-10 – 2024-02-12",
        "2024-02-10 – 2024-02-15",
        "Summary länger (6 Tage erwartet)",
    )
]

    for header_tp, summary_tp, label in cases:
        print("\n======================================")
        print(f"Case: {label}")
        print(f"HEADER : {header_tp}")
        print(f"SUMMARY: {summary_tp}")
        result = compare_time_periods_with_llm(header_tp, summary_tp)
        print("Result vom LLM:", result.model_dump())

if __name__ == "__main__":
    run_manual_tests()

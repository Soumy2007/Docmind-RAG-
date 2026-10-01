"""Generate the sample HR policy PDF used for demos and evaluation.

Run:  python scripts/make_sample_pdf.py
"""
from pathlib import Path

try:
    import pymupdf as fitz
except ImportError:
    import fitz

OUT = Path(__file__).resolve().parent.parent / "data" / "sample_docs" / "Acme_HR_Policy.pdf"

PAGES = [
    [
        ("h", "Acme Technologies - Employee Policy Handbook"),
        ("h2", "1. Leave Policy"),
        ("p", "1.1 Casual leave. Every full-time employee receives 12 days of casual leave per calendar year. Casual leave cannot be carried forward to the next year."),
        ("p", "1.2 Earned leave. Employees earn 1.5 days of earned leave per month, which is 18 days per year. Up to 30 days of earned leave can be carried forward."),
        ("p", "1.3 Sick leave. Employees get 10 days of paid sick leave per year. A medical certificate is required if sick leave exceeds 2 consecutive days."),
        ("p", "1.4 Parental leave. Maternity leave is 26 weeks of paid leave. Paternity leave is 10 working days."),
        ("p", "1.5 Applying for leave. Employees must apply through the HR portal at least 3 working days in advance, except in emergencies."),
    ],
    [
        ("h2", "2. Work From Home"),
        ("p", "2.1 Eligibility. Employees who have completed the 6 month probation may work from home up to 2 days per week with manager approval."),
        ("p", "2.2 Core hours. Employees working from home must be available online between 11:00 AM and 4:00 PM IST."),
        ("p", "2.3 Equipment. The company provides a laptop. Employees on permanent work from home receive a one-time home office allowance of INR 15,000."),
        ("p", "2.4 Internet. Broadband costs are reimbursed up to INR 1,000 per month."),
        ("h2", "3. Working Hours and Attendance"),
        ("p", "3.1 Office timings are 9:30 AM to 6:30 PM, Monday to Friday, including a one hour lunch break."),
        ("p", "3.2 Arriving late more than 3 times in a month results in a half day of leave being deducted."),
    ],
    [
        ("h2", "4. Expense Reimbursement"),
        ("p", "4.1 Travel. Domestic flights must be economy class. Hotel stays are reimbursed up to INR 6,000 per night in metro cities and INR 4,000 per night in other cities."),
        ("p", "4.2 Daily allowance. Employees on outstation travel receive INR 1,200 per day."),
        ("p", "4.3 Claims. Expense claims must be submitted within 30 days with original receipts. Claims above INR 10,000 need director approval."),
        ("h2", "5. Information Security"),
        ("p", "5.1 Passwords. Passwords must have a minimum of 12 characters and be changed every 90 days. Multi-factor authentication is mandatory for email and VPN."),
        ("p", "5.2 Devices. Personal USB drives are prohibited. A lost laptop must be reported to the IT team within 2 hours."),
        ("p", "5.3 Data. Confidential data must never be shared through personal email or messaging apps."),
    ],
    [
        ("h2", "6. Employment Terms and Conduct"),
        ("p", "6.1 Probation and notice. The probation period is 6 months and the notice period during probation is 15 days. After confirmation the notice period is 60 days."),
        ("p", "6.2 Performance reviews. Performance reviews are held twice a year, in April and October."),
        ("p", "6.3 Anti-harassment. The company has zero tolerance for harassment. Complaints can be sent to the Internal Committee at ic@acme.example."),
        ("p", "6.4 Gifts. Employees must not accept gifts worth more than INR 2,000 from vendors or clients."),
    ],
]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    for blocks in PAGES:
        page = doc.new_page(width=595, height=842)
        y = 60.0
        for kind, text in blocks:
            font, size, gap = {"h": ("hebo", 18, 16), "h2": ("hebo", 13, 8), "p": ("helv", 11, 10)}[kind]
            rect = fitz.Rect(56, y, 539, y + 400)
            leftover = page.insert_textbox(rect, text, fontsize=size, fontname=font)
            y += (400 - leftover) + gap
    doc.save(OUT)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()

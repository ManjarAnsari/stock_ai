import smtplib
import os
from email.message import EmailMessage

def send_email(subject, content, to):
    try:
        EMAIL_ADDRESS = os.getenv("ALERT_EMAIL", "youremail@example.com")
        EMAIL_PASSWORD = os.getenv("ALERT_PASSWORD", "your_app_password")

        msg = EmailMessage()
        msg.set_content(content)
        msg['Subject'] = subject
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = to

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)

    except Exception as e:
        print(f"[Email Error] Failed to send email: {e}")
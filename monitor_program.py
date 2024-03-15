import subprocess
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import socket
import os

"""
Monitor the program and check its status: running | broken
The program is running using screen
Once broken, an e-mail is send to the mail box
"""

def get_screen_pid():
    output = subprocess.run(['screen', '-ls'], capture_output=True, text=True)
    lines = output.stdout.split('\n')
    pids = []
    my_pid = os.getpid()
    for line in lines:
        if '.' in line:
            parts = line.split('.')
            pid = parts[0].strip()
            if pid.isdigit():
                pid_num = int(pid)
                if pid_num != my_pid - 1:
                    pids.append(int(pid))
    return pids

def send_email_notification(ip_address, broken_pid):
    sender_email = "hellomonde@icloud.com"
    receiver_email = "wran21@m.fudan.edu.cn"
    password = "rcmw-ypca-vzeq-ujmr"

    message = MIMEMultipart("alternative")
    message["Subject"] = "Screen session broken notification"
    message["From"] = sender_email
    message["To"] = receiver_email

    text = f"Screen session with PID {broken_pid} is broken on {ip_address} at {time.strftime('%Y-%m-%d %H:%M:%S')}."

    part1 = MIMEText(text, "plain")
    message.attach(part1)

    with smtplib.SMTP("smtp.mail.me.com", 587) as server:
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message.as_string())

def main():
    ip_address = socket.gethostbyname(socket.gethostname())
    all_pids = get_screen_pid()
    print(all_pids)
    if len(all_pids) == 0:
        return
    while True:
        current_pids = get_screen_pid()
        if len(all_pids) == len(current_pids) and len(current_pids) > 0:
            print("All PIDs are running, [#NUM {}] ...".format(len(all_pids)))
            time.sleep(5)  # Adjust sleep time as needed
        else:
            broken_pids = set(all_pids).difference(set(current_pids))
            broken_pids = list(broken_pids)
            broken_pids = [str(item) for item in broken_pids]
            pid = '-'.join(broken_pids)
            print("Screen session not found, sending email notification.")
            send_email_notification(ip_address, pid)
            break  # You might want to handle this differently based on your requirements

if __name__ == "__main__":
    main()

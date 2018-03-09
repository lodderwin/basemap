import os
import sys
import smtplib
import yaml
import datetime as dt
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


COMMASPACE = ', '
# Load gmail accpunt settings from yml
gm = yaml.load(open('./gmail_settings.yml'))
# Define time now
now = dt.datetime.now().strftime("%d-%m-%Y %H:%M")

#%%
def send_mail(recipients=['machineearning2018@gmail.com'], 
             subject=None,
             attachments=[], 
             body=None):
    """
    Function to send emails using gmail account.
    
    Parameters
    --------
    recipients : email recipients
    subject : email subject
    attachments : attcahments to attach to email, parse as file location, list
    body : text to include in message, str
    """
    
    # Define sender and password from yaml file
    sender = gm['account']
    pswrd = gm['password']
    
    # Create the enclosing (outer) message
    msg = MIMEMultipart()
    # Add subeject to email if subject is not None
    if subject is not None:
        msg['Subject'] = subject
    msg['To'] = COMMASPACE.join(recipients)
    msg['From'] = sender
    # Add message to email if body is not None
    if body is not None:
        msg.attach(MIMEText(body, 'plain'))

    # Add the attachments to the message
    for file in attachments:
        try:
            with open(file, 'rb') as fp:
                msg = MIMEBase('application', "octet-stream")
                msg.set_payload(fp.read())
            encoders.encode_base64(msg)
            msg.add_header('Content-Disposition', 'attachment', 
                           filename=os.path.basename(file))
            msg.attach(msg)
        except:
            print("Unable to open one of the attachments. Error: ", sys.exc_info()[0])
            raise

    composed = msg.as_string()

    # Send the email
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as s:
            s.ehlo()
            s.starttls()
            s.ehlo()
            s.login(sender, pswrd)
            s.sendmail(sender, recipients, composed)
            s.close()
        print("email sent")
    except:
        print("unable to send the email. error: ", sys.exc_info()[0])
        raise
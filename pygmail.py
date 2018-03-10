import pandas as pd
import numpy as np
import os
import sys
import smtplib
import yaml
import datetime as dt
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# email commaspace
COMMASPACE = ', '
# Load gmail accpunt settings from yml
gm = yaml.load(open('./configs/gmail_settings.yml'))
# Define time now
now = dt.datetime.now().strftime("%d-%m-%Y %H:%M")
# Define today
today = dt.datetime.now().strftime("%d-%m-%Y")
# List of recipients
recipients = ['machineearning2018@gmail.com', 'erwin.lodder@gmail.com',
              'howardriddiough@gmail.com']

def send_mail(recipients=recipients, 
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
    outer = MIMEMultipart()
    
    # Add subeject to email if subject is not None
    if subject is not None:
        outer['Subject'] = subject
    outer['To'] = COMMASPACE.join(recipients)
    outer['From'] = sender
    
    # Add message to email if body is not None
    if body is not None:
        outer.attach(MIMEText(body, 'plain'))

    # Add the attachments to the message
    for file in attachments:
        try:
            with open(file, 'rb') as fp:
                msg = MIMEBase('application', "octet-stream")
                msg.set_payload(fp.read())
            encoders.encode_base64(msg)
            msg.add_header('Content-Disposition', 'attachment', 
                           filename=os.path.basename(file))
            outer.attach(msg)
        except:
            print("Unable to open one of the attachments. Error: ", sys.exc_info()[0])
            raise

    composed = outer.as_string()

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
        
def compose_email(expected_deltas : dict):
    """
    """
    # Create email subject
    subject = 'Expected Deltas Notification: ' + now
    
    # Create data Frame from dict
    df = pd.DataFrame(list(expected_deltas.items()), 
                      columns=['ticker', 'exp_delta'])
    # Sort values by exp_delta
    df = df.sort_values('exp_delta', ascending=False)
    df = df.reset_index(drop=True)
    # Round expected deltas
    df['exp_delta'] = np.round(df.exp_delta, 3)
    # Create String from email
    body = 'Expected Deltas:'
    
    for idx, row in df.iterrows():
        # If idx is one digit then add a space before
        if len(str(idx + 1)) == 1:
            num = ' ' + str(idx + 1)
        else:
            num = str(idx + 1)
            
        # If ticker is three letters long add a space before
        if len(row.ticker) == 3:
            tick = ' ' + row.ticker
        else:
            tick = row.ticker
            
        # If expected delta is postive add a space before
        if row.exp_delta >= 0:
            delta = ' ' + str(row.exp_delta)
        else:
            delta = str(row.exp_delta)
            
        # Create email body string
        body = body + '\n' + num + '. ' + tick + ': ' + delta
        
    # Create list of attachments ranked by expected delta
    files = os.listdir('./plots/' + today)
    # List of tickers in dictionary
    tickers_in_dict = list(expected_deltas.keys())
    
    # Create list of graphs that are only in tickers_in_dict list
    graphs = [file for file in files if file[:4] in tickers_in_dict or file[:3] in tickers_in_dict]
    # Select graph file if it is a compare_predictions or current_predictions
    graphs = [file for file in graphs if file[-22:] == 'current_prediction.png' or file[-15:] == 'predictions.png']
    
    # Use pandas to sort list of graph names by expected delta
    df_graphs = pd.DataFrame(graphs, columns=['filename'])
    # Create ticker id in DataFrame
    df_graphs['ticker'] = df_graphs.filename.str[:4].str.replace('_','')
    # Use file to create graph type column
    df_graphs['graph_type'] = df_graphs.filename.str[-15:]
    
    # Merge df and df_graphs
    df_graphs = df_graphs.merge(df, on='ticker', how='inner')
    # Sort df_graphs by expected delta
    df_graphs = df_graphs.sort_values(['exp_delta','graph_type'])
    
    # Create list of attachments
    attachments = df_graphs.filename.tolist()
    attachments = ['./plots/' + today + '/' + attachment for attachment in attachments]
    
        
    return subject, body, attachments

"""
# Create email body
subject, body, attachments = compose_email(expected_deltas=invest_dict)

# Send email
send_mail(subject=subject,
          attachments=attachments,
          body=body)
"""


#to use google access, make it less secure https://www.google.com/settings/security/lesssecureapps

import smtplib
import email.utils
from email.mime.text import MIMEText
import getpass

# Create the message
msg = MIMEText('This is the body of the message.')
msg['To'] = email.utils.formataddr(('Recipient', 'ndas1971@gmail.com'))
msg['From'] = email.utils.formataddr(('Author', 'ndas1971@gmail.com'))
msg['Subject'] = 'Simple test message'


server = smtplib.SMTP('smtp.gmail.com', 587)  
server.set_debuglevel(True) # show communication with the server
try:
    # identify ourselves, prompting server for supported features
	server.ehlo()
	# If we can encrypt this session, do it
	#if server.has_extn('STARTTLS'):
	server.starttls()
	#server.ehlo() # re-identify ourselves over TLS connection
	
	p = getpass.getpass()
	server.login('ndas1971@gmail.com', p)
	server.sendmail('ndas1971@gmail.com', ['ndas1971@gmail.com',], msg.as_string())
finally:
    server.quit()

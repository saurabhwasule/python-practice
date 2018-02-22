
import getpass, poplib
import email
from io import StringIO   # for py2.x cStringIO
from email.generator import Generator


user = 'ndas1971@gmail.com' 

Mailbox = poplib.POP3_SSL('pop.gmail.com', '995') 
Mailbox.user(user) 

p = getpass.getpass()

Mailbox.pass_(p) 
numMessages = len(Mailbox.list()[1])

#(numMsgs, totalSize) = Mailbox.stat()


print(" No of messages=%s" % numMessages )

for i in range(5):
	msg = Mailbox.retr(i+1)[1]
	f_msg = email.message_from_bytes(b"\n".join(msg))   #in py2.x, use _string 
	for header in [ 'subject', 'to', 'from' ]:
		print('%-8s: %s' % (header.upper(), f_msg[header]))
	print("\n")

# complete message

num = int(input("Input msg # for seeing complete message>"))
msg = Mailbox.retr(num+1)[1]
c_msg = email.message_from_bytes(b"\n".join(msg))  #in py2.x, use _string 
fp = StringIO()
g = Generator(fp, mangle_from_=False, maxheaderlen=60)
g.flatten(c_msg)
text = fp.getvalue()
print(text)

Mailbox.quit()


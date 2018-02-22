import sys 
#first code 
#a = sys.argv[1]
#b = sys.argv[2]
#print(int(a) + int(b))

s = "Hello World"
#H - 1
#e - 1
#l - 3
#
#Take each(ch) from s
#    initialize counter 
#    Take each(ch1) from s
#        if ch and ch1 are same  
#            increment counter 
#    print ch and counter 
d = {}       
for ch in s:
    cnt = 0
    for ch1 in s:
        if ch == ch1:
            cnt = cnt + 1
    d[ch] = cnt 
    print(ch,cnt)
    
    
d = {}
for ch in s:
    if ch not in d: 
        d[ch] = 1
    else: 
        d[ch] = d[ch] + 1 
print(d)



##############################
[]    list - Indexing, IO, Duplicates, Mutable
()        tuple - Immutable, ----- same----
    
{}    set -  Indexing-NA, IO-NA, Duplicates-NA , Mutable
    frozenset - Immutable, ---same----
--------------------------------
dict - key : value , key is set 

--------------
l = [1,2,3,4,5]
res = [1,9,25]

create empty list 
newList would be strip s with '[]' and then split with ','
Take each element(e) from newlist  
        convert e to int  
        and append that  to emptylist 
    

Open a file 
Read lines into a var 
transform var into a new list of tuples - comprehension
    each tuple (line number, length)
sort based length 

         

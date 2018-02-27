import os
import glob
import csv
from xlsxwriter.workbook import Workbook

workbook = Workbook('Test.xlsx')
for ext in ['csv','sql']:
    for csvfile in glob.glob(os.path.join('.', '*.'+ext)):
        filename=csvfile.replace(".\\","")
        extension=filename.split(".")[1]
        worksheet = workbook.add_worksheet(os.path.splitext(filename)[0])  # worksheet with csv file name
        with open(csvfile, 'rt',encoding='utf8') as f:
            if extension=='csv':
                reader = csv.reader(f,delimiter=',')
            else:
                reader = csv.reader(f, delimiter='#')
            for r, row in enumerate(reader):
                for c, col in enumerate(row):
                    worksheet.write(r, c, col)  # write the csv file content into it
workbook.close()

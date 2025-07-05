from openpyxl import Workbook

workbook = Workbook('data.xlsx')
sheet = workbook.active

for row in sheet.iter_rows(values_only=True):
    print(row)

workbook.close

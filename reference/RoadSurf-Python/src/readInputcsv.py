# -*- coding: utf-8 -*-
"""
!MIT License
!Copyright (c) 2026 FMI Open Development

Simple function for reading input
"""
import csv
from datetime import datetime

def read_csv_data(file_path):   
    
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)  # Read the first row as headers
        data= {header:[] for header in headers}
        for row in reader:
            data["time"].append(datetime.strptime(row[0],"%Y%m%d%H"))
            
            for i in range(1,len(headers)):
                data[headers[i]].append(float(row[i]))
    
    return data

# -*- coding: utf-8 -*-
"""
!MIT License
!Copyright (c) 2026 FMI Open Development

Function for writing output to file
"""
import csv

def write_to_csv(output_arrays, input_data, output_file, intervals_in_minutes):
    # Open the CSV file for writing
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write the header
        header = ['Timestamp', 'TsurfOut', 'SnowOut', 'WaterOut', 'IceOut', 'DepositOut', 'Ice2Out']
        writer.writerow(header)

        # Write data at full minutes
        for i in range(0, len(input_data.timestamp)):
            # Check if the timestamp is at a full minute
            if input_data.time[i].minute % intervals_in_minutes == 0 and input_data.time[i].second == 0:
                row = [
                    input_data.time[i].strftime("%Y%m%d%H%M"),
                    round(output_arrays.TsurfOut[i], 1),
                    round(output_arrays.SnowOut[i], 2),
                    round(output_arrays.WaterOut[i], 2),
                    round(output_arrays.IceOut[i], 2),
                    round(output_arrays.DepositOut[i], 2),
                    round(output_arrays.Ice2Out[i], 2)
                ]
                writer.writerow(row)
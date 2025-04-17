#!/bin/bash

for filename in *.pdf; do
	num_pages=$(pdfinfo "$filename" | awk '/Pages:/ {print $2}')
	base="${filename%.pdf}"

	for ((i = 1; i <= num_pages; i++)); do
		out_file="${base}-page-${i}.svg"
		if [ ! -f "$out_file" ]; then
			echo "$out_file does not exist"
			pdf2svg "$filename" "$out_file" "$i"
		fi
	done
done

#!/bin/bash

for filename in *.pdf; do
	num_pages=$(pdfinfo "$filename" | awk '/Pages:/ {print $2}')
	base="${filename%.pdf}"

	pdftoppm -png -r 300 "$filename" "$base"

done

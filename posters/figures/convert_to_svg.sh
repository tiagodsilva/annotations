for filename in *.pdf; do
	pdf2svg $filename ${filename%.pdf}.svg
done 

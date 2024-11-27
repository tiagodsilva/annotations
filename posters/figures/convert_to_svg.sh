for filename in *.pdf; do
	if [ ! -f  ${filename%.pdf}.svg ]; then 
		echo "${filename%.pdf}.svg does not exist"
		pdf2svg $filename ${filename%.pdf}.svg
	fi 
done 

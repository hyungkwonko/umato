#!/usr/bin/env bash

# This is preprocessing file for amazon-review dataset
# https://s3.amazonaws.com/amazon-reviews-pds/tsv/index.txt
# INPUT: amazon review dataset
# OUTPUT: filted amazon review only contain "us_Books" for fasttext trainning

data_dir=$1
output_file=$2
process_tsv() {
  awk -F '\t' '{print "__label__"$8 " " $13 " " $14}' | \
  tr '[:upper:]' '[:lower:]' | \
  sed -e "s/'/ ' /g" -e 's/"//g' -e 's/\./ \. /g' -e 's/<br \/>/ /g' \
      -e 's/,/ , /g' -e 's/(/ ( /g' -e 's/)/ ) /g' -e 's/\!/ \! /g' \
      -e 's/\?/ \? /g' -e 's/\;/ /g' -e 's/\:/ /g' -e 's/<\/s>//g' | tr -s " "
  # Note: we eliminate EOS '</s>' above
}
rm ${output_file}.tmp
for filename in ${data_dir}/tsv/amazon_reviews_us_Books_*.tsv.gz
do
  echo "processing ${filename}"
  gzip -d ${filename} -c | pv | process_tsv >> ${output_file}.tmp
done
mv ${output_file}.tmp ${output_file}


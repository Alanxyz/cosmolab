#!/bin/bash

read -r line 
for n in $line; do
	[ ! -e dat/skymap-$n.fits ] && \
		echo "Downloading hitpix $n" && 
		wget -P dat/ "https://data.desi.lbl.gov/public/dr1/target/skyhealpixs/v1/skymap-$n.fits.gz" && \
		gzip -d dat/skymap-$n.fits.gz
done


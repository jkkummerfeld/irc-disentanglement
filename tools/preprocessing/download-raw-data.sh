#!/bin/bash

echo "Downloading all available data for 2004-2019 from the ubuntu channel"

###for name in ubuntu kubuntu ubuntu-devel "ubuntu+1" ; do
for name in ubuntu ; do
  for year in `seq 2004 2019` ; do
    for month in `seq 1 12` ; do
      max_days="31"
      case "$month" in
        "4" )
          max_days="30" ;;
        "6" )
          max_days="30" ;;
        "9" )
          max_days="30" ;;
        "11" )
          max_days="30" ;;
        "2" )
          case "$year" in
            "2004" )
              max_days="29" ;;
            "2008" )
              max_days="29" ;;
            "2012" )
              max_days="29" ;;
            "2016" )
              max_days="29" ;;
            *)
              max_days="28" ;;
          esac ;;
        *)
          max_days="31" ;;
      esac

      if [ "$month" -lt "10" ] ; then
        month="0$month"
      fi

      mkdir -p $name/$year/$month
      for day in `seq 1 $max_days` ; do
        if [ "$day" -lt "10" ] ; then
          day="0$day"
        fi
        wget https://irclogs.ubuntu.com/$year/$month/$day/%23$name.txt -O $name/$year/$month/${year}-${month}-${day}-$name.txt
      done
    done
  done
done


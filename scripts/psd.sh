#!/bin/sh

#INPUT="Input/DA203598.jpg"
#EXTIMG="results/extracted.png"
#OUTPUT="results/output.psd"

helpFunction()
{
   echo "In PSD Generation"
   echo "Usage: $0 -a parameterA -b parameterB -c parameterC"
   exit 1 # Exit script after printing help
}

mainFunc()
{
	if gimp -i -c -b '(layers-to-psd (list "'$parameterA'" "'$parameterB'") "'$parameterC'")' -b '(gimp-quit 0)' && exit 1; then
      echo "success"
   else
      echo "epic fail"
      exit 0
   fi
}

while getopts "a:b:c:" opt
do
   case "$opt" in
      a ) parameterA="$OPTARG" ;;
      b ) parameterB="$OPTARG" ;;
      c ) parameterC="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Print helpFunction in case parameters are empty
if [ -z "$parameterA" ] || [ -z "$parameterB" ] || [ -z "$parameterC" ]
then
   echo "Some or all of the parameters are empty";
   helpFunction
fi

# Begin script in case all parameters are correct
mainFunc

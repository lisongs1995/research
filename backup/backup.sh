#!/bin/bash
#author: lisongsong02
#description: project config and backup
#usage: bash bakcup.sh 

set -o errexit 
set -o pipefail
#set -x
trap "echo \$LINENO has occured err !!!" ERR

CURRENT=$(cd $(dirname $BASH_SOURCE) && pwd -P)
BASE_DIR="${CURRENT%/*}"
BASELINE="$BASE_DIR/baseline"
DATA="$BASE_DIR/Datasets"

function recoverBaseline(){

  local dilof_git="https://github.com/ngs00/DILOF.git"
  local milof_git="https://github.com/dingwentao/MILOF.git"
  cd $BASELINE && echo "checking baseline.."
  if [ ! -d MILOF ];then
    echo "recovery dilof begin..."
    git clone $dilof_git > /dev/null 2>&1  && echo "recovery dilof ending..."
  fi
  if [ ! -d DILOF ];then
    echo "recovery milof begin..."
    git clone $milof_git > /dev/null 2>&1 && echo "recovery milof ending..."
  fi  
  return $?
}

function download(){

  [ $# -ne 2 ] && {
    echo "wring args given"
    exit 1
  }
  local subset=$1
  local percent10=$2
  command="data=fetch_kddcup99(\"$subset\", percent10=$percent10)"
  echo $command
  return $?
}

function recoverData(){
  import1="from sklearn.datasets import fetch_kddcup99"
  import2="import numpy as np"
  [ -d $DATA ] && cd $DATA 
  echo "checking data"
  for subset in smtp http;do
    for percent10 in True False;do
      if [ ! -s "$subset-$percent10.npy" ];then
        download_data=`download $subset $percent10`
        save="np.save("\"$subset-$percent10\"", data)"
        python3 -c "$import1;$import2;$download_data;$save" &>/dev/null
      fi
    done
  done
  echo "checking data ending"
  return $?
}

function backup(){

  cd ${BASE_DIR%/*} && echo "backup  begining"
  DATE=`date +%Y%m%d`
  [ -d research ] && tar -cvzf "research-$DATE.tar.gz" research &> /dev/null
  [ -s "research-$DATE.tar.gz" ] && cp "research-$DATA.tar.gz" research/backup
  [ -s "research-$DATE.tar.gz" ] && mv "research-$DATA.tar.gz" /media/lss #天有不测风云
  echo "backup ending.."
  return $?
}

function main(){
  whoisme=$1
  [ -z "$whoisme" ] && whoisme="lisongsong02"
  recoverBaseline
  recoverData
  backup
  cd $BASE_DIR
  git add -A
  git commit -m "\[`date +%Y-%m-%d`: backup.sh\]: commited by $whoisme "
  return $?
}

main

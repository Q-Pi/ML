#!/bin/bash

ping 127.0.0.1 -c 4
echo ""

echo "wget 127.0.0.1"
echo ""
wget 127.0.0.1
echo ""

echo "curl 127.0.0.1"
echo ""
wget curl 127.0.0.1
echo ""

rm *html*
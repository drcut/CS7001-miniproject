#!/bin/bash
rm generate_trace/zeusmp_dummy.*
make
./halo ../physical_addr/zeusmp.trace >> generate_trace/zeusmp_dummy.log
cd generate_trace
python translate.py >> zeusmp_dummy.trace
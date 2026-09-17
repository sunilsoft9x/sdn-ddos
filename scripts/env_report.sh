#!/bin/bash
out=data/environment_$(hostname).txt
{ date; git rev-parse HEAD; lsb_release -d; uname -r
  ovs-vsctl --version | head -1; mn --version 2>/dev/null
  java -version 2>&1 | head -1; ls ~/odl/system/org/opendaylight 2>/dev/null | head -1
  ~/venv/bin/python -c 'import sys,torch,numpy,gymnasium;print(sys.version.split()[0],torch.__version__,numpy.__version__,gymnasium.__version__)'
} > "$out" 2>&1; cat "$out"

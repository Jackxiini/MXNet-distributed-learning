import os
import subprocess

server_env = os.environ.copy()
server_env.update({
  "DMLC_ROLE": "server", # Could be "scheduler", "worker" or "server"
  "DMLC_PS_ROOT_URI": "10.157.6.183", # IP address of a scheduler
  "DMLC_PS_ROOT_PORT": "8000", # Port of a scheduler
  "DMLC_NUM_SERVER": "1", # Number of servers in cluster
  "DMLC_NUM_WORKER": "2", # Number of workers in cluster
  "PS_VERBOSE": "2" # Debug mode
})

subprocess.Popen("python -c 'import mxnet as mx'", shell=True, env=server_env)

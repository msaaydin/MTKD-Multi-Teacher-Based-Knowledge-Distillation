
import os
import time
import random
import numpy as np
import cv2
import torch

""" Seeding the randomness. """
def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

""" Create a directory. """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

""" Calculate the time taken """
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs







"""

from flask import Flask, Response
from prometheus_client import start_http_server, Counter, generate_latest, Gauge
import requests
import time
import json
import psutil
import re
import logging
from datetime import datetime


# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S')

app = Flask(__name__)

BASE_URL = "https://desisteely.com"
USERNAME = "abdullah.eid@stu.fsm.edu.tr"
PASSWORD = "1"
COM_TOKEN = "ds13da59-d8e6-49a4-98a2-4df1c9a4bf28"
headers = {
            "Content-Type": "application/json",
            "com-token":COM_TOKEN,
            "Authorization":""
        }
TOKEN = ""

# Define a Prometheus gauge metric for the status code
STATUS_CODE_COUNTER  = Counter('api_status_code', 'Number of API responses by status code', ['status_code',"url","msg","time"])

# Create gauges for CPU and RAM usage with labels for process names
TOP_CPU_USAGE_GAUGE = Gauge('top_cpu_usage', 'CPU usage percentage of the top processes', ['process_name'])
TOP_RAM_USAGE_GAUGE = Gauge('top_ram_usage', 'RAM usage in MB of the top processes', ['process_name'])

# Gauges for total RAM and CPU usage
TOTAL_RAM_GAUGE = Gauge('total_ram', 'Total RAM in MB')
USED_RAM_GAUGE = Gauge('used_ram', 'Used RAM in MB')
FREE_RAM_GAUGE = Gauge('free_ram', 'Free RAM in MB')
TOTAL_CPU_USAGE_GAUGE = Gauge('total_cpu_usage', 'Total CPU usage percentage')


def login(apiUrl = "api/web/v1/login"):
    global TOKEN  # Declare TOKEN as global
    url = f"{BASE_URL}/{apiUrl}"
    payload = {
        "userLoginText": USERNAME,
        "password": PASSWORD
    }
    loginHeaders = {
        "Content-Type": "application/json",
        "com-token":COM_TOKEN
    }

    try:
        # Make the API request with a timeout
        response = requests.post(url, data=json.dumps(payload), headers=loginHeaders, timeout=10)
        
        # Increment the counter based on the status code
        if response.status_code in [500, 400, 404, 403]:
            # Format the time as YY:MM:DD HH:mm:İİ
            current_time = datetime.now().strftime("%y:%m:%d %H:%M:%S")
            STATUS_CODE_COUNTER.labels(status_code=str(response.status_code),time=current_time, url =url, msg =json.loads(response.content)['message']  ).inc()
            print(f"response: {json.loads(response.content)['message']}")
        else:
            # Regular expression to find the JWT in the Set-Cookie header
            jwt_pattern = r"jwt=([^;]+)"
            jwt_token = ""
            # Find JWT using regex
            jwt_match = re.search(jwt_pattern, response.headers["Set-Cookie"])
            if jwt_match:
                jwt_token = jwt_match.group(1)
                headers['Authorization'] = jwt_token;
                print("JWT Token:", jwt_token)
            else:
                print("JWT Token not found.")
            TOKEN = json.loads(response.text)["data"]["webUserToken"]
            
    except requests.Timeout:
        # Handle timeout exception
        print("Request timed out.")
        STATUS_CODE_COUNTER.labels(status_code='timeout').inc()
    except requests.RequestException as e:
        # Handle other request-related exceptions
        print(f"An error occurred: {e}, response: {json.loads(response.content)['message']}")

'''
def alarmArmDisarm(opration = "arm"):
    # opration : arm/disarm
    url = f"{BASE_URL}/api/web/v1/alarm/midline-wifi/{opration}"
    payload = {
        "webUserToken": TOKEN,
        "idAlarmUser": 82,
        "silence": 1
    }
    try:
        # Make the API request with a timeout
        response = requests.post(url, data=json.dumps(payload), headers=headers, timeout=10)
        if response.status_code in [500, 400, 404, 403]:
            # Format the time as YY:MM:DD HH:mm:İİ
            current_time = datetime.now().strftime("%y:%m:%d %H:%M:%S")
            STATUS_CODE_COUNTER.labels(status_code=str(response.status_code),time=current_time, url =url, msg =json.loads(response.content)['message']  ).inc()
            print(f"response: {json.loads(response.content)['message']}")
        print("")       
    except requests.Timeout:
        # Handle timeout exception
        print("Request timed out.")
        STATUS_CODE_COUNTER.labels(status_code='timeout').inc()
    except requests.RequestException as e:
        # Handle other request-related exceptions
        print(f"An error occurred: {e}")


def alarmPanicOrStop(opration = "panic"):
    # opration : panic/stop
    url = f"{BASE_URL}/api/web/v1/alarm/midline-wifi/{opration}"
    payload = {
        "webUserToken": TOKEN,
        "idAlarmUser": 82
    }
    try:
        # Make the API request with a timeout
        response = requests.post(url, data=json.dumps(payload), headers=headers, timeout=10)
        if response.status_code in [500, 400, 404, 403]:
            # Format the time as YY:MM:DD HH:mm:İİ
            current_time = datetime.now().strftime("%y:%m:%d %H:%M:%S")
            STATUS_CODE_COUNTER.labels(status_code=str(response.status_code),time=current_time, url =url, msg =json.loads(response.content)['message']  ).inc()
            print(f"response: {json.loads(response.content)['message']}")
        print("")       
    except requests.Timeout:
        # Handle timeout exception
        print("Request timed out.")
        STATUS_CODE_COUNTER.labels(status_code='timeout').inc()
    except requests.RequestException as e:
        # Handle other request-related exceptions
        print(f"An error occurred: {e}")
'''


def make_request(url, payload):
    try:
        # Make the API request with a timeout
        response = requests.post(url, data=json.dumps(payload), headers=headers, timeout=10)
        
        message = json.loads(response.content).get('message', 'No message')
        if response.status_code in [500, 400, 404, 403]:
            # Format the time as YY:MM:DD HH:mm:İİ
            current_time = datetime.now().strftime("%y:%m:%d %H:%M:%S")
            STATUS_CODE_COUNTER.labels(status_code=str(response.status_code), time=current_time, url=url, msg=message).inc()
            print(f"response: {message}")
        print(f"{url} ({response.status_code}) -> {message} ")       
        logging.info(f"{url} ({response.status_code}) -> {message} ")       

    except requests.Timeout:
        # Handle timeout exception
        print("Request timed out.")
        STATUS_CODE_COUNTER.labels(status_code='timeout', time=current_time, url=url, msg='Timeout').inc()
    except requests.RequestException as e:
        # Handle other request-related exceptions
        print(f"An error occurred: {e}")
        STATUS_CODE_COUNTER.labels(status_code='error', time=current_time, url=url, msg=str(e)).inc()

def alarmArmDisarm(opration="arm"):
    url = f"{BASE_URL}/api/web/v1/alarm/midline-wifi/{opration}"
    payload = {
        "webUserToken": TOKEN,
        "idAlarmUser": 82,
        "silence": 1
    }
    make_request(url, payload)

def alarmPanicOrStop(opration="panic"):
    url = f"{BASE_URL}/api/web/v1/alarm/midline-wifi/{opration}"
    payload = {
        "webUserToken": TOKEN,
        "idAlarmUser": 82
    }
    make_request(url, payload)


def lockUnlockSmartLock(opration="lock"):
    url = f"{BASE_URL}/api/web/v1/smart-lock/utopic-r/{opration}"
    payload = {
        "webUserToken": TOKEN,
        "idSmartLockUser": 82
    }
    make_request(url, payload)


def update_top_process_metrics():
    # Get all processes
    myprocesses=[]
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info']):
        try:
            proc_info = proc.info
            # Append process information with CPU and RAM usage
            myprocesses.append({
                'name': proc_info['name'],
                'cpu_percent': proc_info['cpu_percent'],
                'ram_mb': proc_info['memory_info'].rss / (1024 * 1024)  # Convert to MB
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    # Sort processes by CPU usage and RAM usage
    top_cpu_processes = sorted(myprocesses, key=lambda x: x['cpu_percent'], reverse=True)[:5]
    top_ram_processes = sorted(myprocesses, key=lambda x: x['ram_mb'], reverse=True)[:5]

    # Clear previous metrics
    TOP_CPU_USAGE_GAUGE.clear()
    TOP_RAM_USAGE_GAUGE.clear()

    # Update metrics for top CPU usage processes
    for proc in top_cpu_processes:
        TOP_CPU_USAGE_GAUGE.labels(process_name=proc['name']).set(proc['cpu_percent'])
    # Update metrics for top RAM usage processes
    for proc in top_ram_processes:
        TOP_RAM_USAGE_GAUGE.labels(process_name=proc['name']).set(proc['ram_mb'])

def update_system_metrics():
    # Get memory usage statistics
    virtual_mem = psutil.virtual_memory()
    total_ram = virtual_mem.total / (1024 * 1024)  # Convert to MB
    used_ram = virtual_mem.used / (1024 * 1024)  # Convert to MB
    free_ram = virtual_mem.available / (1024 * 1024)  # Convert to MB

    # Update Prometheus gauges for RAM usage
    TOTAL_RAM_GAUGE.set(total_ram)
    USED_RAM_GAUGE.set(used_ram)
    FREE_RAM_GAUGE.set(free_ram)

    # Get total CPU usage
    total_cpu_usage = psutil.cpu_percent(interval=1)  # Percentage over a second

    # Update Prometheus gauge for CPU usage
    TOTAL_CPU_USAGE_GAUGE.set(total_cpu_usage)


@app.route('/metrics', methods=['GET'])
def metrics():
	# Return metrics in the format Prometheus expects
	return Response(generate_latest(), content_type="text/plain; version=0.0.4; charset=utf-8")
if __name__ == '__main__':
    start_http_server(9200)  # Exposes metrics on http://localhost:9200/metrics
    while True:
        login()
        time.sleep(3)
        alarmArmDisarm()
        time.sleep(3)
        alarmArmDisarm(opration="disarm")
        time.sleep(3)
        alarmPanicOrStop()
        time.sleep(3)
        alarmPanicOrStop(opration="stop")
        update_system_metrics()
        update_top_process_metrics()

		#time.sleep(60)  # Check the API every 60 seconds

"""
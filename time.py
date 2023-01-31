from datetime import datetime
try:
    while True:
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
except KeyboardInterrupt:
    pass

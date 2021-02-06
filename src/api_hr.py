from flask import Flask, request, g
import os
import json
import time
import hr_process
from common import utils
import sys
import traceback

def initRouteWithPrefix(route_function, prefix='', mask='{0}{1}'):
    '''
      Defines a new route function with a prefix.
      The mask argument is a `format string` formatted with, in that order:
        prefix, route
    '''

    def newroute(route, *args, **kwargs):
        '''New function to prefix the route'''
        return route_function(mask.format(prefix, route), *args, **kwargs)

    return newroute


app = Flask(__name__)

@app.before_request
def before_request():
    g.request_start_time = time.time()


@app.teardown_request
def teardown_request(exception=None):
    print("From before_request to teardown_request: %.2fms" % ((time.time() - g.request_start_time) * 1000))


@app.route("/", defaults={"path": ""}, methods=['GET', 'POST'])
@app.route("/<path:path>", methods=['GET', 'POST'])
def do_upload_file(path):
    if request.method == 'GET':
        return "hello, world"
    if request.method == 'POST':
        try:
            utils.AppConfig.write_log(request, False)
            config = request.get_json()
            utils.AppConfig.write_log(config, False)

            ################################################
            #TODO: call your predictive function here
            ################################################

            return None #TODO: write your return the result here

        except Exception as inst:
            error = "Error:"
            traceback_str = ''.join(traceback.format_tb(inst.__traceback__))
            error += traceback_str
            utils.AppConfig.write_log(traceback_str, True)
            utils.AppConfig.write_log(inst, True)
            data = {
                'Status': error
            }
            return json.dumps(data, ensure_ascii=False)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8099, threaded=True)

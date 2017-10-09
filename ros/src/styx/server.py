#!/usr/bin/env python

import socketio
import eventlet
import eventlet.wsgi
import time
import rospy
from flask import Flask, render_template

from bridge import Bridge
from conf import conf

sio = socketio.Server()
app = Flask(__name__)
msgs = []

dbw_enable = False

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)

def send(topic, data):
    s = 1
    msgs.append((topic, data))
    #sio.emit(topic, data=json.dumps(data), skip_sid=True)

bridge = Bridge(conf, send)

@sio.on('telemetry')
def telemetry(sid, data):
    rospy.loginfo("telemetry msg")
    global dbw_enable
    if data["dbw_enable"] != dbw_enable:
        dbw_enable = data["dbw_enable"]
        bridge.publish_dbw_status(dbw_enable)
    bridge.publish_odometry(data)
    for i in range(len(msgs)):
        topic, data = msgs.pop(0)
        sio.emit(topic, data=data, skip_sid=True)

@sio.on('control')
def control(sid, data):
    rospy.loginfo("control msg")
    bridge.publish_controls(data)

@sio.on('obstacle')
def obstacle(sid, data):
    rospy.loginfo("obstacle msg")
    bridge.publish_obstacles(data)

@sio.on('lidar')
def obstacle(sid, data):
    rospy.loginfo("lidar msg")
    bridge.publish_lidar(data)

@sio.on('trafficlights')
def trafficlights(sid, data):
    rospy.loginfo("trafficlights msg")
    bridge.publish_traffic(data)

@sio.on('image')
def image(sid, data):
    rospy.loginfo("image msg")
    bridge.publish_camera(data)

if __name__ == '__main__':

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

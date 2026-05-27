[app]
title = Apple Detector
package.name = appledetector
package.domain = org.example
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,onnx
version = 1.0
requirements = python3,kivy,opencv-python,numpy,pillow,onnxruntime
orientation = portrait
android.permissions = INTERNET, WRITE_EXTERNAL_STORAGE, READ_EXTERNAL_STORAGE
android.api = 30
android.minapi = 24
android.ndk = 25b
android.sdk = /usr/local/lib/android/sdk
android.ndk_path = /usr/local/lib/android/sdk/ndk/27.3.13750724
android.accept_sdk_license = True
p4a.branch = master

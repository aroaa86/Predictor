[app]

# (str) Title of your application
title = Preditcor

# (str) Package name
package.name = predictor

# (str) Package domain (needed for android/ios packaging)
package.domain = org.adilson

# (str) Source code where main.py live
source.dir = .

# (str) Main Python file
source.main = main.py

# (list) List of source files to include
source.include_exts = py,kv,png,jpg,jpeg,atlas,json,ttf,db

# (list) List of inclusions using pattern matching
#source.include_patterns = assets/*,images/*

# (str) Application version
version = 1.0

# (list) Application requirements
requirements = python3,kivy==2.3.1,kivymd==2.0.0,materialyoucolor==3.0.4,materialshapes==0.3,asynckivy==0.6.4,asyncgui==0.6.3,pillow

# (str) Supported orientation
orientation = landscape

# (bool) Fullscreen mode
fullscreen = 1


#
# OSX Specific
#

# author = © Copyright Info
# license = MIT


#
# Android specific
#

# (bool) Indicate if the application should be fullscreen or not
# fullscreen = 1

# (string) Presplash of the application
# presplash.filename = %(source.dir)s/data/presplash.png

# (string) Icon of the application
# icon.filename = %(source.dir)s/data/icon.png

# (str) Supported architectures
android.archs = arm64-v8a

# (int) Minimum API required
android.minapi = 24

# (int) Android API to use
android.api = 34

# (str) Android NDK version
android.ndk = 27b

# (str) Android SDK version
# android.sdk = 34

# (str) Android entry point
android.entrypoint = org.kivy.android.PythonActivity

# (bool) Enable AndroidX
android.enable_androidx = True

# (bool) Use Android appcompat
android.add_androidx = True

# (str) Android app theme
android.apptheme = @android:style/Theme.Material.Light.NoActionBar

android.accept_sdk_license = True

# (list) List of Java .jar files to add
#android.add_jars = foo.jar

# (list) List of Java files to add
#android.add_src =

# (list) Android AAR files to add
#android.add_aars =

# (list) Gradle dependencies
#android.gradle_dependencies =

# (str) Android NDK path
#android.ndk_path =

# (str) Android SDK path
#android.sdk_path =

# (str) Android entry point
#android.entrypoint = org.kivy.android.PythonActivity

# (str) Android activity class name
#android.activity_class_name = org.kivy.android.PythonActivity

# (str) Android app name
#android.app_name = %(title)s

# (bool) Indicate whether the app should be built in debug mode
android.debug = 1


#
# Python-for-Android specific
#

# (str) Python-for-Android branch to use
#p4a.branch = develop

# (str) Python-for-Android URL
#p4a.url =

# (str) Python-for-Android commit
#p4a.commit =

# (str) Python-for-Android directory
#p4a.source_dir =

# (str) Python-for-Android extra arguments
#p4a.extra_args =


#
# iOS Specific
#

# (str) Name of the certificate
#ios.codesign.allowed = false

# (str) Path to the certificate
#ios.codesign.debug =


#
# Buildozer release
#

# (str) Buildozer log level
log_level = 2

# (str) Warn on deprecated options
warn_on_root = 1

[app]

# (str) Title of your application
title = Predictor

# (str) Package name
package.name = predictor

# (str) Package domain
package.domain = org.adilson

# (str) Source code directory
source.dir = .

# (str) Main Python file
source.main = oraculoGui.py

# (list) Files to include
source.include_exts = py,kv,png,jpg,jpeg,atlas,json,ttf,db,csv,txt

# (str) Application version
version = 1.0

# (list) Application requirements
requirements = python3==3.14.2,hostpython3==3.14.2,kivy==2.3.1,kivymd==2.0.0,materialyoucolor==3.0.4,materialshapes==0.3,asynckivy==0.6.4,asyncgui==0.6.3,pillow

# (str) Supported orientation
orientation = landscape

# (bool) Fullscreen mode
fullscreen = 1

# (str) Application icon
# icon.filename = %(source.dir)s/data/icon.png

# (str) Presplash image
# presplash.filename = %(source.dir)s/data/presplash.png

# (str) Application author
# author = Adilson

# (str) Application license
# license = MIT

# ==========================================================
# Android
# ==========================================================

# (str) Supported architectures
android.archs = arm64-v8a

# (int) Minimum Android API
android.minapi = 24

# (int) Android API
android.api = 36

# (int) Android target API
android.targetapi = 36

# (str) Android NDK version
android.ndk = 29

# (str) Android entry point
android.entrypoint = org.kivy.android.PythonActivity

# (bool) Enable AndroidX
android.enable_androidx = True
android.add_androidx = True

# (str) Android application theme
android.apptheme = @android:style/Theme.Material.Light.NoActionBar

# (bool) Accept Android SDK licenses
android.accept_sdk_license = True

# (bool) Debug build
android.debug = 1

# (list) Android permissions
# android.permissions = INTERNET

# ==========================================================
# Python-for-Android
# ==========================================================

# Use the branch with Python 3.14 support
p4a.branch = develop

# ==========================================================
# Buildozer
# ==========================================================

log_level = 2
warn_on_root = 1

# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['oraculoGui.py'],
    pathex=[],
    binaries=[],
    datas=[('chromedriver.exe', '.'), ('*.kv', '.')],
    hiddenimports=[
        'pkg_resources.py2_warn',  # Mantemos o workaround
        'kivy.app',                # Módulos essenciais do Kivy
        'kivy.factory',            # Módulos essenciais do Kivy
        'kivy.core.text.markup',   # Para resolver problemas de texto/KivyMD
        'kivymd.app',              # Módulos essenciais do KivyMD
        'kivymd.uix'               # Módulos essenciais do KivyMD
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='oraculoGui',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

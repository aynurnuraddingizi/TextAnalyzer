# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['vocabmaster/main.py'],
    pathex=['.', 'vocabmaster'],
    binaries=[],
    datas=[
        ('text_analyzer_icon.ico', '.'),
        ('dict_de.sqlite3', '.'),
        ('dict_tr.sqlite3', '.'),
        ('dict_ar.sqlite3', '.'),
        ('dict_ru.sqlite3', '.'),
        ('dict_es.sqlite3', '.'),
        ('cefr_en.json', '.'),
        ('cefr_es.json', '.'),
        ('phrase_en.json', '.'),
        ('phrase_es.json', '.'),
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tensorflow', 'torch', 'torchvision', 'keras', 'scipy', 'pandas', 'sklearn', 'matplotlib', 'transformers', 'onnx', 'tf2onnx', 'numpy', 'IPython', 'jupyter', 'notebook'],
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
    name='TextAnalyzer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['text_analyzer_icon.ico'],
)

# 本文件禁止修改，需要修改请联系zhenbinye

from kaiwu_agent.utils.conf_parser import IniParser, JsonParser, YamlParser, TreeParser
import glob
import importlib.util
import sys

__doc__ = '自动实例化conf下所有类型的配置文件，实例后的对象名称为[type]_[folder]_[subfolder]_[filename]'

__all__ = []

_excluded_file_list = ['__init__.py', 'metayaml.yaml']
_excluded_folder_list = ['__pycache__', 'template']


def instance_conf_obj_from_spec_type(type_file, caller, base_path=__path__[0], name_prefix=None):
    for file_path in glob.glob('%s/*.%s' % (base_path, type_file)):
        if file_path.strip().split('/')[-1] in _excluded_file_list:
            continue
        name_obj = name_prefix if name_prefix != None else type_file
        name_obj = '%s_%s' % (name_obj, file_path.strip().split('/')[-1].split('.')[0])
        globals()[name_obj] = caller(file_path)
        __all__.append(name_obj)
    for folder_path in glob.glob('%s/**/' % (base_path)):
        folder_path = folder_path[:-1]
        if folder_path.strip().split('/')[-1] in _excluded_folder_list:
            continue
        name_obj = name_prefix if name_prefix != None else type_file
        name_obj = '%s_%s' % (name_obj, folder_path.strip().split('/')[-1]) 
        instance_conf_obj_from_spec_type(type_file, caller, folder_path, name_obj)

def _instance_conf_obj_from_py(config_file_path):
    spec = importlib.util.spec_from_file_location("module.name", config_file_path)
    foo = importlib.util.module_from_spec(spec)
    sys.modules["module.name"] = foo
    spec.loader.exec_module(foo)
    return foo

instance_conf_obj_from_spec_type('yaml', YamlParser)
instance_conf_obj_from_spec_type('ini', IniParser)
instance_conf_obj_from_spec_type('json', JsonParser)
instance_conf_obj_from_spec_type('tree', TreeParser)
instance_conf_obj_from_spec_type('py', _instance_conf_obj_from_py)

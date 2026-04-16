# 本文件禁止修改，需要修改请联系zhenbinye

import configparser
import json
import argparse
from kaiwu_agent.utils.common_func import wrapped_dict, unwrapped_dict, is_number_repl_isdigit
import yaml
import os
from types import MethodType


class BaseParser(object):
    def __init__(self, config_file_path) -> None:
        self.config_file_path = config_file_path
    
    def __getattr__(self, attr):
        return getattr(self.dict_obj, attr)

    def __getitem__(self, n):
        return self.dict_obj[n]
    
    def convert_to_raw_dict(self):
        if hasattr(self, 'dict_obj'):
            return unwrapped_dict(self.dict_obj)
        else:
            raise NotImplemented
    

class IniParser(BaseParser):
    def __init__(self, config_file_path) -> None:
        super().__init__(config_file_path)
        self.parser = configparser.ConfigParser()
        self.parser.read(self.config_file_path)
        self.flag_section = (True, None)
    
    def get(self, section, option):
        """
        get option: raw method
        """
        return self.parser.get(section, option)

    def __getattr__(self, attr):
        """
        get option: parser.section.option        
        """
        if self.flag_section[0] and self.parser.has_section(attr):
            self.flag_section = (False, attr)
            return self
        if not self.flag_section[0] and self.parser.has_option(self.flag_section[1], attr):
            ret = self.parser.get(self.flag_section[1], attr)
            self.flag_section = (True, None)
            if is_number_repl_isdigit(ret):
                return float(ret) if '.' in ret else int(ret) 
            return ret
        raise AttributeError('\'IniParser1\' object has no attribute \'%s\'' % attr)
    
    def __str__(self) -> str:
        return str(self.parser._sections)    
    

class JsonParser(BaseParser):
    def __init__(self, config_file_path) -> None:
        super().__init__(config_file_path)
        with open(self.config_file_path, "r") as file_obj:
            self.dict_obj = wrapped_dict(json.loads(file_obj.read()))

    def __getitem__(self, n):
        return self.dict_obj[n]

class YamlParser(BaseParser):
    def __init__(self, config_file_path) -> None:
        super().__init__(config_file_path)
        with open(self.config_file_path) as file_obj:
            self.dict_obj = wrapped_dict(yaml.full_load(file_obj))
        
    def __iter__(self):
        return iter(self.dict_obj)
    
    def __getitem__(self, n):
        return self.dict_obj[n]
    
    def render_config_from_file(self, yaml_file_path):
        with open(yaml_file_path, "r") as file_obj:
            yaml_source = yaml.full_load(file_obj)
        self.render_config_from_dict(yaml_source)
    
    def render_config_from_dict(self, yaml_source):
        tmp = self.convert_to_raw_dict()
        tmp.update(yaml_source)
        self.dict_obj = wrapped_dict(tmp)
    
    def dump_config_to_file(self, yaml_file_path):
        tmp = self.convert_to_raw_dict()
        with open(yaml_file_path, 'w') as fp:
            yaml.dump(tmp, fp)


class TreeParser(BaseParser):
    def __init__(self, config_file_path) -> None:
        super().__init__(config_file_path)
        with open(self.config_file_path) as file_obj:
            self.dict_obj = dict()
            lines = list()
            for line in file_obj.readlines():
                if len(line.strip()) > 0:
                    lines.append(line)
            
        self.config_file_name = os.path.basename(config_file_path)

        dict_obj = self.__build_dict_from_list(lines)
        self.dict_obj = wrapped_dict(dict_obj)

        if self.config_file_name == 'switch.tree':

            cls_pre = 'Switch_'
            self.create_class_tree_from_dict_tree('root', object, self.root, cls_pre)

            def fn(self, str_args):
                cls_args = getattr(self, cls_pre + str_args)
                flag = True
                while cls_args is not object:
                    flag = flag and cls_args.VALUE
                    cls_args = cls_args.super_cls
                return flag
            self.check = MethodType(fn, self)

            def fn2(self, str_args, value):
                cls_args = getattr(self, cls_pre + str_args)
                cls_args.VALUE = value
            self.set = MethodType(fn2, self)
    
    def create_class_tree_from_dict_tree(self, root_name, super_cls, sub_tree, cls_pre):
        root_name = cls_pre + root_name
        setattr(self, root_name, type(root_name, (object, ), {}))
        getattr(self, root_name).VALUE = sub_tree.VALUE
        getattr(self, root_name).super_cls = super_cls

        if len(sub_tree.keys()) == 1:    
            return

        sub_names = []
        for key in sub_tree.keys():
            if key != 'VALUE':
                sub_names.append(key)

        getattr(self, root_name).sub_cls = []
        for sub_name in sub_names:
            self.create_class_tree_from_dict_tree(sub_name, getattr(self, root_name), sub_tree[sub_name], cls_pre)
            getattr(self, root_name).sub_cls.append(getattr(self, cls_pre + sub_name))

    def __build_dict_from_list(self, lines):
        
        list_idx = list()
        for idx, line in enumerate(lines):
            if line[0] != ' ':
                list_idx.append(idx)
        
        dict_obj = dict()
        for idx, begin in enumerate(list_idx):
            if idx+1 < len(list_idx):
                list_sub = lines[begin: list_idx[idx+1]]
            else:
                list_sub = lines[begin:]

            key, value = self.__parse_line_to_dict(list_sub[0])
            
            tmp = [i[2:] for i in list_sub[1:]]

            dict_sub = self.__build_dict_from_list(tmp)
            dict_sub['VALUE'] = value.strip()
            if self.config_file_name == "switch.tree":
                 dict_sub['VALUE'] = True if value.strip() == 'true' else False 

            dict_obj[key] = dict_sub 
            
        return dict_obj
            
    
    def __parse_line_to_dict(self, line):
        """
            "    use_nature: true " to {'nature': True}
        """
        k, v = line.replace(" ", "").split(":")
        return k, v


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=False)
    parser.add_argument('--name', default="JsonParser", help='')
    args = parser.parse_args()

    from kaiwu_agent.utils import conf_parser
    cls = getattr(conf_parser, args.name)
    
    if 'ConfParser' == args.name:
        file_path = '../conf/demo.ini'
        parser = cls(file_path)
        print(parser.main.name)
        print(parser.arena.time)
    if 'JsonParser' == args.name:
        file_path = '../conf/demo.json'
        parser = cls(file_path)
        print(parser.main.name)
        print(parser.arena.time)
    if 'YamlParser' == args.name:
        file_path = '../conf/demo.yaml'
        parser = cls(file_path)
        print(parser.main.name)
        print(parser.arena.time)
    
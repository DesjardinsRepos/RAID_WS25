import re
import codecs
from typing import List

# Keywords set - C and C++ keywords up to C11 and C++17
keywords = frozenset({'__asm', '__builtin', '__cdecl', '__declspec', '__except', '__export', '__far16', '__far32',
                      '__fastcall', '__finally', '__import', '__inline', '__int16', '__int32', '__int64', '__int8',
                      '__leave', '__optlink', '__packed', '__pascal', '__stdcall', '__system', '__thread', '__try',
                      '__unaligned', '_asm', '_Builtin', '_Cdecl', '_declspec', '_except', '_Export', '_Far16',
                      '_Far32', '_Fastcall', '_finally', '_Import', '_inline', '_int16', '_int32', '_int64',
                      '_int8', '_leave', '_Optlink', '_Packed', '_Pascal', '_stdcall', '_System', '_try', 'alignas',
                      'alignof', 'and', 'and_eq', 'asm', 'auto', 'bitand', 'bitor', 'bool', 'break', 'case',
                      'catch', 'char', 'char16_t', 'char32_t', 'class', 'compl', 'const', 'const_cast', 'constexpr',
                      'continue', 'decltype', 'default', 'delete', 'do', 'double', 'dynamic_cast', 'else', 'enum',
                      'explicit', 'export', 'extern', 'false', 'final', 'float', 'for', 'friend', 'goto', 'if',
                      'inline', 'int', 'long', 'mutable', 'namespace', 'new', 'noexcept', 'not', 'not_eq', 'nullptr',
                      'operator', 'or', 'or_eq', 'override', 'private', 'protected', 'public', 'register',
                      'reinterpret_cast', 'return', 'short', 'signed', 'sizeof', 'static', 'static_assert',
                      'static_cast', 'struct', 'switch', 'template', 'this', 'thread_local', 'throw', 'true', 'try',
                      'typedef', 'typeid', 'typename', 'union', 'unsigned', 'using', 'virtual', 'void', 'volatile',
                      'wchar_t', 'while', 'xor', 'xor_eq', 'NULL'})

# Main function names
main_set = frozenset({'main'})

# Main function arguments
main_args = frozenset({'argc', 'argv'})

# Operators
operators3 = {'<<=', '>>='}
operators2 = {
    '->', '++', '--', '**',
    '!~', '<<', '>>', '<=', '>=',
    '==', '!=', '&&', '||', '+=',
    '-=', '*=', '/=', '%=', '&=', '^=', '|='
}
operators1 = {
    '(', ')', '[', ']', '.',
    '+', '&',
    '%', '<', '>', '^', '|',
    '=', ',', '?', ':',
    '{', '}', '!', '~'
}

def to_regex(lst):
    return r'|'.join([f"({re.escape(el)})" for el in lst])

regex_split_operators = to_regex(operators3) + to_regex(operators2) + to_regex(operators1)

def clean_gadget(gadget):
    """
    Clean C/C++ code by replacing user variables with standardized names.
    
    Args:
        gadget: List of code lines
        
    Returns:
        List of cleaned code lines
    """
    # dictionary to map function name to symbol name + number
    fun_symbols = {}
    # dictionary to map variable name to symbol name + number
    var_symbols = {}

    fun_count = 1
    var_count = 1

    # regular expression to find function name candidates
    rx_fun = re.compile(r'\b([_A-Za-z]\w*)\b(?=\s*\()')
    # regular expression to find variable name candidates
    rx_var = re.compile(r'\b([_A-Za-z]\w*)\b((?!\s*\**\w+))(?!\s*\()')

    # final cleaned gadget output to return
    cleaned_gadget = []

    for line in gadget:
        # replace any non-ASCII characters with empty string
        ascii_line = re.sub(r'[^\x00-\x7f]', r'', line)
        # remove all hexadecimal literals
        hex_line = re.sub(r'0[xX][0-9a-fA-F]+', "HEX", ascii_line)
        # get all function and variable matches
        user_fun = rx_fun.findall(hex_line)
        user_var = rx_var.findall(hex_line)

        # process function names
        for fun_name in user_fun:
            if len({fun_name}.difference(main_set)) != 0 and len({fun_name}.difference(keywords)) != 0:
                # check if function name already in dictionary
                if fun_name not in fun_symbols.keys():
                    fun_symbols[fun_name] = 'FUN' + str(fun_count)
                    fun_count += 1
                # ensure only function name gets replaced
                hex_line = re.sub(r'\b(' + fun_name + r')\b(?=\s*\()', fun_symbols[fun_name], hex_line)

        # process variable names
        for var_name in user_var:
            if len({var_name[0]}.difference(keywords)) != 0 and len({var_name[0]}.difference(main_args)) != 0:
                # check if variable name already in dictionary
                if var_name[0] not in var_symbols.keys():
                    var_symbols[var_name[0]] = 'VAR' + str(var_count)
                    var_count += 1
                # ensure only variable name gets replaced
                hex_line = re.sub(r'\b(' + var_name[0] + r')\b(?:(?=\s*\w+\()|(?!\s*\w+))(?!\s*\()',
                                var_symbols[var_name[0]], hex_line)

        cleaned_gadget.append(hex_line)
    return cleaned_gadget

def tokenize_code(code):
    """
    Tokenize C/C++ code into a list of tokens.
    
    Args:
        code: String containing C/C++ code
        
    Returns:
        List of tokens
    """
    gadget = []
    tokenized = []
    
    # remove all string literals
    no_str_lit_line = re.sub(r'["]([^"\\\n]|\\.|\\\n)*["]', '', code)
    # remove all character literals
    no_char_lit_line = re.sub(r"'.*?'", "", no_str_lit_line)
    code = no_char_lit_line

    # Split code into lines and create gadget
    for line in code.splitlines():
        if line == '':
            continue
        stripped = line.strip()
        gadget.append(stripped)

    # Clean the gadget
    clean = clean_gadget(gadget)

    # Process each line in the cleaned gadget
    for cg in clean:
        if cg == '':
            continue

        # Remove code comments
        pat = re.compile(r'(/\*([^*]|(\*+[^*\/]))*\*+\/)|(\/\/.*)')
        cg = re.sub(pat, '', cg)

        # Remove newlines & tabs
        cg = re.sub('(\n)|(\\\\n)|(\\\\)|(\\t)|(\\r)', '', cg)
        
        # Split on operators and spaces
        splitter = r' +|' + regex_split_operators + r'|(\/)|(\;)|(\-)|(\*)'
        cg = re.split(splitter, cg)

        # Remove None type and empty strings
        cg = list(filter(None, cg))
        cg = list(filter(str.strip, cg))
        
        # Add tokens to result list
        tokenized.extend(cg)

    return tokenized


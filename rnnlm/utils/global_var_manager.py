class GlobalVarManager:
    """
    Class for managing global states
    """

    def __init__(self, name_to_value_dict=None):
        self.vars = dict()
        self.original_values = dict()
        if name_to_value_dict is not None:
            self._vars_to_dict(name_to_value_dict)

    def add(self, name_to_value_dict):
        self._vars_to_dict(name_to_value_dict)

    def update(self, name_to_new_value_dict):
        for name in name_to_new_value_dict.keys():
            if name not in self.vars.keys():
                raise ValueError("Can't update var {} which does not exist. Not updating anything".format(name))

        for name, new_val in name_to_new_value_dict.items():
            self.vars[name] = new_val

    def remove(self, var_names):
        for var in var_names:
            if var not in self.vars.keys():
                raise ValueError("Can't remove var {} which does not exists".format(var))
            self.vars.pop(var)
            self.original_values.pop(var)

    def reset_all(self):
        for k, v in self.original_values.items():
            self.vars[k] = v

    def reset_one(self, var):
        if var not in self.vars.keys():
            raise ValueError("Cannot reset var {} since it does not exist".format(var))
        value = self.original_values[var]
        self.vars[var] = value

    def _vars_to_dict(self, name_to_value_dict):
        for name, value in name_to_value_dict.items():
            if name in self.vars.keys():
                raise ValueError("GlobalVarManager has already var {}, use update function to update value of "
                                 "existing var".format(name))
            self.vars[name] = value
            self.original_values[name] = value


'''
* Attributes:
    + Class Attribute: Associated with the class itself and shared among all instances.
    + Instance Attribute: Associated with a particular instance of the class.

* Methods:
    + Instance Method: A method that operates on an instance of the class.
    + Class Method: A method that operates on the class itself and can modify class 
    state which applies to all instances.
    + Static Method: A method that is bound to the class but does not influence the 
    class state.
'''

class ClassTool:
    @staticmethod
    def add_method(cls, method_name, method):
        setattr(cls, method_name, method)
    
    @staticmethod
    def add_attribute(cls, attr_name, initialize=None):
        '''
        This method is used for adding class attribute
        '''
        setattr(cls, attr_name, initialize)

    @staticmethod
    def add_attribute(obj, attr_name, initialize=None):
        '''
        This method is used for adding instance attribute
        '''
        setattr(obj, attr_name, initialize)
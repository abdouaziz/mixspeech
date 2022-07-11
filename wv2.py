from unicodedata import name


class Personn :
    def __init__(self , name , age , gender):
        self.__name=name
        self.__age =age
        self.__gender=gender


    @property
    def Name(self):
        return self.__name

    @Name.setter
    def Name(self, value):
        self.__name=name

    def mymethod(self):
        return "je suis la boy"



if __name__=="__main__":
    p = Personn("abdou",32,"M")
    print(p.Name)
    p.Name="diop"
    print(p.mymethod)
    print(p.mymethod())
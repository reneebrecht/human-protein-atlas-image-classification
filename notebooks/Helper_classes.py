class Location_in_Target(object):
    ''' Class to get a list of name of pictures which are labeled with a specific 
        location and the same number of names of pictures which labels do not 
        contain the specific location.
        ...
        Attributes:
        ----------
        location: int 
            number which define a location in a human cell
        pictures: panda.DataFrame
            list of the names of the pictures labeled with the location and 
            same number of pictures without the label
        
        Methods:
        ----------
        save_pictures(pictures, column_name):
            Saves the list of picture names in an Attribute
        get_pictures()
            Returns the list of picture names
        determine_pictures(labels)  
            Find pictures with and without the specific label and call save method
            when list is build
    '''

    def __init__(self, location): 
        '''Parameters
           ----------
           location: int 
                number which define a location in a human cell
        '''
        import pandas as pd
        
        self.location = location
        self.pictures = pd.DataFrame()

    def save_pictures(self, pictures, column_name):
        ''' Saves the list of pictures in the attribute pictures

            Parameters
            ----------
            pictures: panda.DataFrame
                list of the names of the pictures labeled with the location and 
                same number of pictures without the label
            column_name: str
                Description of the data saved in the DataFrame used as column names 
        '''

        self.pictures[column_name] = pictures

    def get_pictures(self):
        '''Gives back the list of picture names which are labeled or not labeled
            with the given location
        '''

        return self.pictures

    def determine_pictures(self, labels):
        ''' First the names of the pictures labeled with the given location are 
            saved in a list and the method to save it in a DataFrame column is 
            called. 
            The same number of names from pictures not labeled with the location 
            are chosen randomly. The saving function is called with it again. 

            Parameters
            ----------
            labels: DataFrame with all picture names and Targets       
        '''
        import numpy as np
        
        pictures = [label.Id  for index, label in labels.iterrows() 
                              if str(self.location) in label.Target]
        self.save_pictures(pictures, f'pictures_with_location_{self.location}')
        #The number of pictures labeled with the given location is needed to get the 
        #same number of pictures not labeled with the location
        number_needed_pictures= len(pictures)
        pictures = [label.Id  for index, label in labels.iterrows() 
                              if str(self.location) not in label.Target]
        self.save_pictures(np.random.choice(pictures, number_needed_pictures),
                                f'pictures_without_location_{self.location}')

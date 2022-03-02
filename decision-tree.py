import numpy as np

class Node():
    def __init__(self, children = None, threshold = None, 
                       fea_index = None, kind_of_splitt = None, label_HP = None ):
        """
        Contractor for a Decision Tree leaf.  
        Parameters
        ----------
        children        : ndarray
                          it is a 2-D array, it contains all childrennodes of a node.
                          A child is structed as [children, self.feeatrue_index, Predicted lable or None]. 
        threshold       : float
                          Is the value for splitting the continuous features. 
        fea_index       : int 
                          Is the index of the used feature for splitting the node.
        kind_of_splitt  : String
                          Was the splitt done on continuous or categorical data.
        label_HP        : Int
                          Is the label with highest probability, witch is used if no missing value is 
                          encountered.   
        """
        self.children = children 
        self.threshold = threshold
        self.feature_index = fea_index
        self.kind_of_splitt = kind_of_splitt
        self.label_HP = label_HP 
                        


class DT():
    def __init__(self, number_of_target_label, 
                       threshold_for_attribut_classification):
        """
        Contractor for the Decision Tree.  
        Parameters
        ----------
        number_of_target_label                  : int
                                                  Contians the number of unique used class labels of the dataset. 
        threshold_for_attribut_classification   : int
                                                  Contains the threshold for classifing a numerical feature as categorical. 
        """
        self.num_tlable = number_of_target_label
        self.thr_classification = threshold_for_attribut_classification
    

    def fit (self, X, y):
        """
        Saves the trainings- and testdata.  
        Parameters
        ----------
        X   : ndarray
              Contains the features. 
        y   : ndarray
              Contains the Class label
        """
        self.X_train = X
        self.Y_train = y
    

    def kind_of_attribut(self, x):
        """
        Labels the coloums of the imput as categorical or continuous.
        All coloums are marked as categorical data, if they contain a string 
        value or contain <= than in self.thr_classification specified unique values.
        An categorical coloum is marked as False and a continuous coloum as True.
        Parameters
        ----------
        x : ndarray
        Returns
        -------
        kind_of_features : array_like
                           1-D specifies, if coloum is categorical (False) or 
                           continuous (True) of the imput. 
        """
        return [False   if num_diff_values_in_column <= self.thr_classification 
                    else True 
                        for num_diff_values_in_column in [len(np.unique(column)) 
                            if isinstance(column[column == column][0], str) != True 
                            else False  for column in x.T]]

    def choose_best_random_label(self, y):
        """
        Returns class label, with the highest probability. 
        Parameters
        ----------
        y        : array_like
                   Contains the class labels.  
        Returns
        -------
        label_HP : int
                   Is the class label with highest probability.  
        """
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]


    def built_tree(self):
        """
        Calls the function that builds the decision tree. Determines whether the features 
        are continuous or categorical. 
        Returns
        -------
        dt : (Nodeobject, np.array([bool]))
              Nodeobject, is the root of the dt, and 1-D array specifies, if the features are classified as categorical (False) or 
              continuous (True). 
        """
        root = Node(None, None, None)
        kind_of_features = self.kind_of_attribut(self.X_train)
        all_x_continuous  = self.X_train[:,kind_of_features]
        all_x_categorical = self.X_train[:,np.invert(kind_of_features)]



        def help_built_tree(x_categorical, x_continuous, all_y, parent_node):
            """
            Builds the dt, searches for the best splitt in continuous and categorical data (for the training data). 
            In the case of a split, the training data is distributed accordingly and new child/ren node/s are created.
            These nodes are splitt until only one class label remains or one sampel -> leaf that holds this value is created.   
            Parameters
            ----------
            x_categorical : ndarray
                            Contains left categorical features, after splitt. 
            x_continuous  : ndarray
                            Contains left continuous features, after splitt.
            all_y         : ndarray
                            All left class attributes, after splitt. 
            parent_node   : Node object
                            Node, for splitting the features. 
            """
            best_splitt_con = self.continuous_splitt(x_categorical , x_continuous, all_y)
            best_splitt_cate = self.categorical_splitt(x_categorical, x_continuous, all_y)


            if best_splitt_cate[0] > best_splitt_con[0]:
                index_of_best_feature, values, splitts = best_splitt_cate[1], best_splitt_cate[2], best_splitt_cate[3:]
                parent_node.kind_of_splitt = "categorical"
                parent_node.label_HP = self.choose_best_random_label(all_y)
                parent_node.feature_index = index_of_best_feature 
                parent_node.children = np.array([None, None, None]*len(values)).reshape((len(values), 3))
                i = 0

                for  value, x_cate, x_con, y in splitts:
                    y_values = np.unique(y)
                    if len(y) == 1 or len(y_values) == 1:
                        parent_node.children[i, :] = [None, value, y[0]]
                    
                    elif np.shape(x_cate)[1] <= 1:
                        values, counts = np.unique(y, return_counts=True)
                        ind = np.argmax(counts)
                        parent_node.children[i, :] = [None, value, values[ind]]

                    else:
                        new_parentnode = Node()
                        parent_node.children[i, :] = [new_parentnode, value, None]
                        help_built_tree(x_cate, x_con, y, new_parentnode)
                    i+=1

            elif best_splitt_con[1] != None:
                _, threshold, feature, all_x_g, x_cate_g, all_y_g, all_x_sEq, x_cate_sEq, all_y_sEq = best_splitt_con
                parent_node.threshold = threshold
                parent_node.feature_index = feature
                parent_node.kind_of_splitt = "continuous"

                if len(all_y_g) == 0 or len(all_y_sEq) == 0:

                    if len(all_y_g) == len(all_y_sEq) == 0:
                        parent_node.children = np.array([[None, feature, self.choose_best_random_label(all_y)]])
                        
                    elif len(all_y_g) == 0:
                        parent_node.children = np.array([[None, feature, all_y_sEq[0]]])
                        
                    else:
                        parent_node.children = np.array([[None, feature, all_y_g[0]]])
                        
                elif np.all(all_y_g == all_y_g[0]) and np.all(all_y_sEq == all_y_sEq[0]): 
                    parent_node.children = np.array([[None, feature, all_y_sEq[0]], [None, feature, all_y_g[0]]])
                    
                elif np.all(all_y_g == all_y_g[0]):
                    new_parentnode = Node()
                    parent_node.children = np.array([[new_parentnode, feature, None], [None, feature, all_y_g[0]]])
                    help_built_tree(x_cate_sEq, all_x_sEq, all_y_sEq, new_parentnode)

                elif np.all(all_y_sEq == all_y_sEq[0]):
                    new_parentnode = Node()
                    parent_node.children = np.array([[None, feature, all_y_sEq[0]], [new_parentnode, feature, None]])
                    help_built_tree(x_cate_g, all_x_g, all_y_g, new_parentnode)
                    
                else:
                    new_parentnode_l = Node()
                    new_parentnode_r = Node()
                    parent_node.children = np.array([ [new_parentnode_l, feature, None] , [new_parentnode_r, feature, None]])
                    help_built_tree(x_cate_sEq, all_x_sEq, all_y_sEq, new_parentnode_l), help_built_tree(x_cate_g, all_x_g, all_y_g, new_parentnode_r)
            
        help_built_tree(all_x_categorical, all_x_continuous, self.Y_train, root)
        return (root, kind_of_features)
    

    def categorical_splitt(self, x_cate, x_con, all_y):
        """
        Finds the highest information gain, for the categorical data.  
        Parameters
        ----------
        x_cate : ndarray 
                 Contains categorical features for splitt.
        x_con  : ndarray
                 Contains continuous features, they are splitt arcording to
                 the best categorical splitt.
        all_y  : ndarray
        Returns
        -------
        categorical_splitt : array-like
                             Contians highest InformationGain and attibutes of the 
                             splitt, with the splitted x_cate, x_con and all_y features.
        """
        if np.shape(x_cate)[1] == 0:
            return [-1]
        
        all_information_Gain = np.array([None]*np.shape(x_cate)[1])
    
        for index, feature in enumerate(x_cate.T):
            values, counts = np.unique(feature, return_counts=True)
            n = len(all_y)
            all_entropies = np.array([None]*len(values))

            for i, value in enumerate(values):
                all_class_attributs_with_y = all_y[ feature == value] 
                _, counts = np.unique(all_class_attributs_with_y, return_counts=True)
                
                if len(counts) <= 1:
                    all_entropies[i] = 0
                else:
                    all_entropies[i] = self.entropy(counts[0], counts[1]) * np.sum(counts)/ n 

            values, counts = np.unique(all_y, return_counts=True)
            overall_Entropy = self.entropy(counts[0], counts[1])
            all_information_Gain[index] = overall_Entropy - np.sum(all_entropies)
        
        index_of_best_feature = np.argmax(all_information_Gain)
        best_splitting_feature = x_cate[:, index_of_best_feature]
        values, counts = np.unique(best_splitting_feature, return_counts=True)
        y_values, y_counts = np.unique(best_splitting_feature, return_counts=True)
        best_splitt = [np.max(all_information_Gain), index_of_best_feature, values]

        if np.shape(x_cate)[1] > 1:
            x_cate = np.delete(x_cate, index_of_best_feature, 1)

        for value, count in zip(values, counts):
            if np.shape(x_cate)[1] >= 1:
                cate_features_for_new_branche = x_cate[best_splitting_feature == value]
                con_features_for_new_branche = x_con[best_splitting_feature == value]
            else:
                cate_features_for_new_branche = []

            if np.shape(x_con)[1] > 1:
                con_features_for_new_branche = x_con[best_splitting_feature == value]
            else:
                con_features_for_new_branche = []

            class_attributs_for_new_branche = all_y[best_splitting_feature == value]
            best_splitt.append([value, cate_features_for_new_branche, con_features_for_new_branche, class_attributs_for_new_branche])
        
        return best_splitt


    def continuous_splitt(self, x_cate, x_con, all_y):
        """
        Finds the highest information gain, for continuous data.  
        Parameters
        ----------
        x_cate : ndarray 
                 Contains categorical features for splitt.
        x_con  : ndarray
                 Contains continuous features, they are splitt arcording to
                 the best categorical splitt.
        all_y  : ndarray
        Returns
        -------
        categorical_splitt : array-like
                             Contians highest InformationGain and attibutes of the 
                             splitt, with the splitted x_cate, x_con and all_y features.
        """
        # inizialize highest_informationGain and best_split for the case no possible splitt ist found, 
        # in this case use the class attribute in all_y[1] as the child node of this branche 
        highest_informationGain, best_split = -1, [-1, None]
    
        # Go through all combinations of the class attributes 
        # e. g. class attibutes {0,1,2} => ({0}, {1,2}) ({1}, {0,2}) ({2}, {1,0})
        for partition in range(self.num_tlable):
            # critiria for a possible splitt: 
            # sort attribute column ascending if the i and the i+1 value have a 
            # different class attribut, calculate the infoGain   
            for feature_index in range(np.shape(x_con)[1]):
                num_of_rows = len(all_y)
                index_to_sort_rows = x_con[:, feature_index].argsort()
                x_sorted = x_con[index_to_sort_rows]
                y_sorted = all_y[index_to_sort_rows]

                # search for best splitt 
                # num_of_rows-2, because with acces the index i+1 
                for i in range(num_of_rows-2):
                    # i and the i+1 value have a different class attribut -> calculate the infoGain
                    if (partition == y_sorted[i] and partition != y_sorted[i+1]) or (partition == y_sorted[i+1] and partition != y_sorted[i]):
                        threshold = (x_sorted[i, feature_index] + x_sorted[i+1, feature_index])/2

                        all_x_g_thr = x_sorted[ x_sorted[: ,feature_index] > threshold]
                        all_y_g_thr = y_sorted[ x_sorted[: ,feature_index] > threshold]

                        all_x_sEq_thr = x_sorted[ x_sorted[: ,feature_index] <= threshold]
                        all_y_sEq_thr = y_sorted[ x_sorted[: ,feature_index] <= threshold]
                         
                        entropy = self.entropy(np.sum(y_sorted == partition), np.sum(y_sorted != partition))
                        sub_entropy_0 = self.entropy(np.sum(all_y_sEq_thr == partition), np.sum(all_y_sEq_thr != partition))
                        sub_entropy_1 = self.entropy(np.sum(all_y_g_thr == partition), np.sum(all_y_g_thr != partition))

                        informationGain = self.information_Gain(entropy, num_of_rows, len(all_y_g_thr), len(all_y_sEq_thr), sub_entropy_0, sub_entropy_1 )
                        
                        if highest_informationGain == None or informationGain > highest_informationGain:
                            highest_informationGain = informationGain
                            x_cate_sorted = x_cate[index_to_sort_rows]
                            all_x_cate_g_thr = x_cate_sorted[ x_sorted[: ,feature_index] > threshold]
                            all_x_cate_sEq_thr = x_cate_sorted[ x_sorted[: ,feature_index] <= threshold]

                            best_split = (highest_informationGain, threshold, feature_index, all_x_g_thr, 
                                          all_x_cate_g_thr, all_y_g_thr, all_x_sEq_thr, all_x_cate_sEq_thr, all_y_sEq_thr)

        return best_split
    

    def entropy(self, p_plus, p_minus):
        """
        Calculates the Entropy.  
        Parameters
        ----------
        p_plus : int
                 Count of values of this categorie. 
        p_minus: int 
                 Count of values of this categorie.
        Returns
        -------
        entropy : float 
        """
        if p_plus == 0 or p_minus == 0 :
            return 0
        else:
            return -(p_plus/ (p_plus+p_minus)) * np.log2(p_plus/ (p_plus+p_minus)) -(p_minus/ (p_plus+p_minus)) * np.log2(p_minus/ (p_plus+p_minus))


    def information_Gain(self,entropy, num_of_rows, num_geater_than_t, num_smaller_equ_as_t, sub_entropy_0, sub_entropy_1):
        """
        Calculates informationGain, only used for splitting continuous features.  
        Parameters
        ----------
        entropy             : float
        num_of_rows         : int 
        num_geater_than_t   : int 
        num_smaller_equ_as_t: int 
        sub_entropy_0       : float
        sub_entropy_1       : float
        Returns
        -------
        information_Gain    : float 
        """
        return entropy - (num_smaller_equ_as_t / num_of_rows) * sub_entropy_0 - (num_geater_than_t/ num_of_rows) * sub_entropy_1

    
    def predict(self, x_test, y_test, root, kind_of_features):
        predicted_labels = [self._predict(x, root, kind_of_features) for x in x_test]
        return 100/ np.shape(y_test)[0] * sum([ 1 for y_test_label, predicted_label in zip (y_test, predicted_labels ) if y_test_label == predicted_label]) 

    def _predict(self, x, root, kind_of_features):
        """
        Predict sampel x with the build dt.  
        Parameters
        ----------
        x                   : ndarray
                              row of the testingdata for prediction. 
        root                : Node-object 
        kind_of_features    : array-like
                              1-D Bool, mark a featurs as continuous or categorical 
        Returns
        -------
        prediction    : int
                        predicted classlabel  
        """
        current_node = root
        x_con = x[kind_of_features]
        x_cate = x[np.invert(kind_of_features)]
        
      
        while current_node != None:
            if len(current_node.children) <= 1 :
                    return  current_node.children[0,2]
            if current_node.kind_of_splitt == "continuous":
                if x_con[current_node.feature_index] <= current_node.threshold:
                    if current_node.children[0,0] == None:
                        return current_node.children[0,2]
                    else: 
                        current_node = current_node.children[0,0]
                else: 
                    if current_node.children[1,0] == None:
                        return current_node.children[1,2]
                    else:
                        current_node = current_node.children[1,0]

            else:
                flag = True
                for child_node, attribute, y  in current_node.children:
                    if x_cate[current_node.feature_index] == attribute:
                        if child_node == None:
                            return y
                        else:
                            flag = False
                            x_cate = np.delete(x_cate, current_node.feature_index, 0)
                            current_node = child_node
                            break; 
                        
                if flag:
                    # Used if missing value in dt accours 
                    return current_node.label_HP
            
### Example ####           
## X_train and y_train needs to be provided 
# def main(DT, Node):
#     dt = DT(2, 10)
#     dt.fit(X_train, y_train) 
    
#     root, kind_of_features = dt.built_tree()
#     print("Accuracy:", dt.predict(X_test, y_test, root, kind_of_features)) 

# if __name__ == "__main__":
#     main(DT, Node)

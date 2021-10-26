#*----------------------------------------------------------------------------*
#* Copyright (C) 2021 Politecnico di Torino, Italy                            *
#* SPDX-License-Identifier: Apache-2.0                                        *
#*                                                                            *
#* Licensed under the Apache License, Version 2.0 (the "License");            *
#* you may not use this file except in compliance with the License.           *
#* You may obtain a copy of the License at                                    *
#*                                                                            *
#* http://www.apache.org/licenses/LICENSE-2.0                                 *
#*                                                                            *
#* Unless required by applicable law or agreed to in writing, software        *
#* distributed under the License is distributed on an "AS IS" BASIS,          *
#* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
#* See the License for the specific language governing permissions and        *
#* limitations under the License.                                             *
#*                                                                            *
#* Author:  Matteo Risso                                                      *
#*----------------------------------------------------------------------------*

import re
import sys

def get_flops(model):
    flops = 0
    for i in range(len(model.layers)):
        layer = model.get_layer(index=i)
        
        if re.search('conv.+', layer.get_config()['name']) : 
            #print (layer)
            
            in_shape = layer.input.get_shape().as_list()[2]
            out_shape = layer.output.get_shape().as_list()[2]
            c_in = layer.input.get_shape().as_list()[3]
            c_out = layer.output.get_shape().as_list()[3]
            k = layer.get_config()['kernel_size'][1]
 
            flops += 2 * out_shape * k * c_in * c_out
            
        elif re.search('dense.+', layer.get_config()['name']) : 
            #print (layer)
            
            out_shape = layer.output.get_shape().as_list()[1]
            in_shape = layer.input.get_shape().as_list()[1]
            # import pdb
            # pdb.set_trace()
            flops += 2 * out_shape * in_shape
            
        elif re.search('batch_normalization.+', layer.get_config()['name']) or \
            re.search('bn.+', layer.get_config()['name']):  
            #print (layer)
            
            # bn after conv
            if len(layer.output.get_shape().as_list()) == 4:
                out_shape = layer.output.get_shape().as_list()[2]
                c_out = layer.output.get_shape().as_list()[3]
                
                flops += 2 * 4 * c_out * out_shape
            # bn after fc
            else:
                out_shape = layer.output.get_shape().as_list()[1]
                
                flops += 2 * 4 * out_shape
                
        elif re.search('average_pooling2d.+', layer.get_config()['name']) or \
            re.search('^pool.+', layer.get_config()['name']): 
            
            #print(layer)
            out_shape = layer.output.get_shape().as_list()[2]
            c_out = layer.output.get_shape().as_list()[3]
            
            flops += 2 * out_shape * c_out
            
        elif re.search('act.+', layer.get_config()['name']) :
            
            # act after conv
            if len(layer.output.get_shape().as_list()) == 4:
                out_shape = layer.output.get_shape().as_list()[2]
                c_out = layer.output.get_shape().as_list()[3]
                
                flops += 2 * out_shape * c_out
            # act after fc
            else:
                out_shape = layer.output.get_shape().as_list()[1]
                
                flops += 2 * 4 * out_shape  
        
        elif re.search('flatten.+', layer.get_config()['name']) :
            pass
        
        elif re.search('drop.+', layer.get_config()['name']) :
            pass
        
        elif re.search('zero_padding2d.+', layer.get_config()['name']) or \
            re.search('pad.+', layer.get_config()['name']):
            pass
        
        elif re.search('global_average_pooling2d.+', layer.get_config()['name']) or \
            re.search('gpool.+', layer.get_config()['name']):
            pass
        
        else:
            print("Unknown layer: {}".format(layer.get_config()['name']))
            sys.exit()
            
    return flops         
            
            
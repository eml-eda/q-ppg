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
#* Author:  Alessio Burrello                                                  *
#*----------------------------------------------------------------------------*

echo "Test TimePPG/TEMPONet quantized: 05/07/2021"

mkdir test_05_07
networks="TempoNetfloat TempoNet_layer_quantized_16 TempoNet_layer_quantized_2 TempoNet_layer_quantized_4 TempoNet_layer_quantized_8 TempoNet_layer_quantized_big TempoNet_layer_quantized_medium TempoNet_layer_quantized_small TimePPG_big_quantized_2 TimePPG_big_quantized_4 TimePPG_big_quantized_8 TimePPG_big_quantized_big TimePPG_big_quantized_medium TimePPG_big_quantized_small TimePPG_medium_quantized_2 TimePPG_medium_quantized_4 TimePPG_medium_quantized_8 TimePPG_medium_quantized_big TimePPG_medium_quantized_medium TimePPG_medium_quantized_small TimePPG_small_quantized_2 TimePPG_small_quantized_4 TimePPG_small_quantized_8 TimePPG_small_quantized_big TimePPG_small_quantized_medium TimePPG_small_quantized_small TimePPGfloat_big TimePPGfloat_medium TimePPGfloat_small"

for net in $networks;
do
	echo $net
	python3 test_TEMPONetDaliaTrainer.py --gpu 0 --cross-validation True -a $net --finetuning True > ./test_05_07/cr_val_$net.txt
done
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
networks="TempoNet_multiprec_big TempoNet_multiprec_medium TempoNet_multiprec_small TimePPG_big_multiprec_big TimePPG_big_multiprec_medium TimePPG_big_multiprec_small TimePPG_medium_multiprec_big TimePPG_medium_multiprec_medium TimePPG_medium_multiprec_small TimePPG_small_multiprec_big TimePPG_small_multiprec_medium TimePPG_small_multiprec_small"

for net in $networks;
do
	echo $net
	python3 test_TEMPONetDaliaTrainer.py --gpu 3 --cross-validation True -a $net --finetuning True > ./test_05_07/cr_val_$net.txt
done
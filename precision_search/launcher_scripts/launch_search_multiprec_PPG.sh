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

echo "Test TimePPG/TEMPONet search mix: 05/07/2021"

networks="mixTempoNet_layer_248_multiprec  mixTimePPG_big_248_multiprec mixTimePPG_medium_248_multiprec  mixTimePPG_small_248_multiprec "
for net in $networks;
do
	echo $net
	echo "cd 000035"
	python3 test_TEMPONetDaliaTrainer.py --gpu 2 --cross-validation False -a $net  --cd 0.000035 > ./test_05_07/cr_val_$net\_000035.txt
done

for net in $networks;
do
	echo $net
	echo "cd 00035"
	python3 test_TEMPONetDaliaTrainer.py --gpu 2 --cross-validation False -a $net  --cd 0.00035 > ./test_05_07/cr_val_$net\_00035.txt
done


for net in $networks;
do
	echo $net
	echo "cd 0035"
	python3 test_TEMPONetDaliaTrainer.py --gpu 2 --cross-validation False -a $net  --cd 0.0035 > ./test_05_07/cr_val_$net\_0035.txt
done

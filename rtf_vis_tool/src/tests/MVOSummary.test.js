/* Unit Test for the MVOSummary.js Component.

Author: Adi Singh
*/
import React from 'react';

// Third-Party Package Imports.
import Enzyme, { shallow } from 'enzyme';
import Adapter from 'enzyme-adapter-react-16';

// Local Imports.
import MVOSummary from '../Components/MVOSummary';

// Configure enzyme for current React version.
Enzyme.configure({ adapter: new Adapter() });

describe('MVOSummary.js Test', () => {

    /* Creates a sample MVOSummary.js component to check if it renders the correct number of p tags.
    */
    it('MVOSummary has 12 <p> tags.', () => {
        const minimal_rot_avg_json = {
            "num_rotations_computed": 12.0,
            "rotation_angle_error_deg": {
                "summary": {
                    "min": 0.03382447361946106,
                    "max": 0.1998642235994339,
                    "median": 0.05946389585733414,
                    "mean": 0.08046150952577591,
                    "stddev": 0.04999322071671486,
                    "quartiles": {
                        "q0": 0.03382447361946106,
                        "q1": 0.043377893045544624,
                        "q2": 0.059463897719979286,
                        "q3": 0.10575252398848534,
                        "q4": 0.1998642235994339
                    }
                }
            }
        };
        
        const minimal_trans_avg_json = {
            "num_total_1dsfm_measurements": 66.0,
            "num_inlier_1dsfm_measurements": 66.0,
            "num_outlier_1dsfm_measurements": 0.0,
            "1dsfm_precision_5_deg": 1.0,
            "1dsfm_recall_5_deg": 1.0,
            "num_translations_estimated": 12.0,
            "1dsfm_inlier_angular_errors_deg": {
                "summary": {
                    "min": 0.021724557504057884,
                    "max": 1.3048033714294434,
                    "median": 0.16076821088790894,
                    "mean": 0.2332172691822052,
                    "stddev": 0.21187157928943634,
                    "quartiles": {
                        "q0": 0.021724557504057884,
                        "q1": 0.12488921545445919,
                        "q2": 0.16076821833848953,
                        "q3": 0.2750871032476425,
                        "q4": 1.3048033714294434
                    }
                }
            },
            "1dsfm_outlier_angular_errors_deg": {
                "summary": {
                    "min": NaN,
                    "max": NaN,
                    "median": NaN,
                    "mean": NaN,
                    "stddev": NaN
                },
                "full_data": []
            },
            "translation_angle_error_deg": {
                "summary": {
                    "min": 0.020061750713985384,
                    "max": 1.121173946411271,
                    "median": 0.18422899027680323,
                    "mean": 0.21783325939037315,
                    "stddev": 0.1730275911864224,
                    "quartiles": {
                        "q0": 0.020061750713985384,
                        "q1": 0.12272492134310564,
                        "q2": 0.18422899027680323,
                        "q3": 0.25304694679999956,
                        "q4": 1.121173946411271
                    }
                }
            },
            "translation_error_distance": {
                "summary": {
                    "min": 0.004873777739703655,
                    "max": 0.0741724967956543,
                    "median": 0.011197246611118317,
                    "mean": 0.02210106886923313,
                    "stddev": 0.022036021575331688,
                    "quartiles": {
                        "q0": 0.004873777739703655,
                        "q1": 0.00856664392631501,
                        "q2": 0.011197247076779604,
                        "q3": 0.02635383326560259,
                        "q4": 0.0741724967956543
                    }
                }
            }
        };
    
        const wrapper = shallow(<MVOSummary rotation_averaging_metrics={minimal_rot_avg_json} translation_averaging_metrics={minimal_trans_avg_json}/>);
        const p_tags = wrapper.find('p');
        
        expect(p_tags.length).toEqual(12);
    })
})
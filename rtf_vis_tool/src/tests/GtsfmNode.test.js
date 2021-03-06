/* Unit Tests for the DivNode.js Component.

Author: Adi Singh
*/
import React from 'react'

// Third-Party Package Imports.
import Enzyme, { shallow } from 'enzyme'
import Adapter from 'enzyme-adapter-react-16'

// Local Imports.
import GtsfmNode from '../Components/GtsfmNode';

// Configure enzyme for current React version.
Enzyme.configure({ adapter: new Adapter() })

describe('GtsfmNode.js Test', () => {

    /*  Passes in sample text and checks if the resulting DivNode component is displaying the proper
        text.
    */
    it('Node displays `Node Testing` when that text is passed into props.', () => {
        const wrapper = shallow(<GtsfmNode text={'Node Testing'}/>);
        const div = wrapper.find('div[id="Node Testing"]');
        expect(div.text()).toEqual('Node Testing');
    });

    /*  Passes in styling properties like textColor, backgroundColor, topOffset, and leftOffset and 
        then checks if the resulting DivNode component is styled properly.
    */
    it('Node has all the correct stylings passed in from props.', () => {
        const wrapper = shallow(<GtsfmNode
                                    textColor={'black'} 
                                    backgroundColor={'gray'} 
                                    topOffset={'10%'} 
                                    leftOffset={'20%'} 
                                    text={'Node Testing'}/>);

        const div = wrapper.find('div[id="Node Testing"]');
        expect(div.props().style.top).toEqual('10%');
        expect(div.props().style.left).toEqual('20%');
        expect(div.props().style.backgroundColor).toEqual('gray');
        expect(div.props().style.color).toEqual('black');
    });
})
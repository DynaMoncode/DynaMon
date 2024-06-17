#ifndef _SUMAX_P4_
#define _SUMAX_P4_

#include <core.p4>
#include <tna.p4>

header ethernet_h {
    bit<48> dst_addr;
    bit<48> src_addr;
    bit<16> ether_type;
}

header ipv4_h {
    bit<4> version;
    bit<4> ihl;
    bit<8> diffserv;
    bit<16> total_len;
    bit<16> identification;
    bit<3> flags;
    bit<13> frag_offset;
    bit<8> ttl;
    bit<8> protocol;
    bit<16> hdr_checksum;
    bit<32> src_addr;
    bit<32> dst_addr;
}

header udp_h {
    bit<16> src_port;
    bit<16> dst_port;
    bit<16> total_len;
    bit<16> checksum;
}

header my_protocol_h { // The header for priority
    bit<8> priority;
}

struct my_ingress_header_t {
    ethernet_h ethernet;
    ipv4_h ipv4;
    udp_h udp;
    my_protocol_h my_protocol;
}

const bit<16> ETHERTYPE_IPV4 = 0x0800;
const bit<8> IPV4PROTOCOL_UDP = 17;

#define BUCKET_ID_BITS 16
#define NUMBER_OF_BUCKETS 1 << BUCKET_ID_BITS

typedef bit<16> bucket_id_t;

struct sumax {
    bit<16>         sum_value;
    bit<16>         max_value;
}

struct id_pair { // For heavy hitter.
    bit<32>         counter;
    bit<32>         id;
}

struct my_ingress_metadata_t {
    bit<1> do_insert;
    bit<1> part_used; // 0 for a, 1 for b
    bit<32> index_1;
    bit<32> index_2;
    bit<32> index_3;
    bucket_id_t insert_base;
    bucket_id_t index_mask;
    bit<16> sum_w;
}

parser IngressParser(packet_in        pkt,
    /* User */
    out my_ingress_header_t           hdr,
    out my_ingress_metadata_t         meta,
    /* Intrinsic */
    out ingress_intrinsic_metadata_t  ig_intr_md) 
{
    state start {
        pkt.extract(ig_intr_md);
        pkt.advance(PORT_METADATA_SIZE);
        transition meta_init;
    }
    state meta_init {
        meta.index_1 = 0;
        meta.index_2 = 0;
        meta.index_3 = 0;
        meta.do_insert = 0;
        meta.insert_base = 0;
        meta.index_mask = NUMBER_OF_BUCKETS - 1;
        meta.sum_w = 65535;
        transition parse_ethernet;
    }
    state parse_ethernet {
        pkt.extract(hdr.ethernet);
        transition select(hdr.ethernet.ether_type) {
            ETHERTYPE_IPV4: parse_ipv4;
            default: accept;
        }
    }
    state parse_ipv4 {
        pkt.extract(hdr.ipv4);
        transition select(hdr.ipv4.protocol) {
            IPV4PROTOCOL_UDP: parse_udp;
            default: accept;
        }
    }
    state parse_udp {
        pkt.extract(hdr.udp);
        transition parse_my_protocol;
    }
    state parse_my_protocol {
        pkt.extract(hdr.my_protocol);
        transition accept;
    }
}

control Ingress(
    /* User */
    inout my_ingress_header_t                        hdr,
    inout my_ingress_metadata_t                      meta,
    /* Intrinsic */
    in    ingress_intrinsic_metadata_t               ig_intr_md,
    in    ingress_intrinsic_metadata_from_parser_t   ig_prsr_md,
    inout ingress_intrinsic_metadata_for_deparser_t  ig_dprsr_md,
    inout ingress_intrinsic_metadata_for_tm_t        ig_tm_md)
{
    
    ////////////////////////////////// switch the part used //////////////////////////////////
    action use_part_a() {
        meta.part_used = 0;
    }

    action use_part_b() {
        meta.part_used = 1;
    }

    table part_switch_table {
        key = {}

        actions = {
            use_part_a;
            use_part_b;
        }

        size = 1;
        default_action = use_part_a;
    }

    ////////////////////////////////// hash //////////////////////////////////
    CRCPolynomial<bit<32>>(0x04C11DB7,false,false,false,32w0xFFFFFFFF,32w0xFFFFFFFF) crc32a;
    CRCPolynomial<bit<32>>(0x741B8CD7,false,false,false,32w0xFFFFFFFF,32w0xFFFFFFFF) crc32b;
    CRCPolynomial<bit<32>>(0xDB710641,false,false,false,32w0xFFFFFFFF,32w0xFFFFFFFF) crc32c;
    Hash<bit<32>>(HashAlgorithm_t.CUSTOM, crc32a) hash_1;
    Hash<bit<32>>(HashAlgorithm_t.CUSTOM, crc32b) hash_2;
    Hash<bit<32>>(HashAlgorithm_t.CUSTOM, crc32c) hash_3;

    action apply_hash_1() {
        meta.index_1 = hash_1.get({hdr.ipv4.src_addr, hdr.ipv4.dst_addr, hdr.udp.src_port, hdr.udp.dst_port});
    }
    action apply_hash_2() {
        meta.index_2 = hash_2.get({hdr.ipv4.src_addr, hdr.ipv4.dst_addr, hdr.udp.src_port, hdr.udp.dst_port});
    }
    action apply_hash_3() {
        meta.index_3 = hash_3.get({hdr.ipv4.src_addr, hdr.ipv4.dst_addr, hdr.udp.src_port, hdr.udp.dst_port});
    }

    ////////////////////////////////// load balance //////////////////////////////////
    action insert() {
        meta.do_insert = 1;
    }

    table lb_table { // See if the flow will be inserted in the current switch.
        key = {
            meta.index_1 : lpm;
        }

        actions = {
            insert;
            NoAction;
        }

        size = 1024;
        const default_action = NoAction;
    }

    ////////////////////////////////// prioirty //////////////////////////////////
    action set_base(bucket_id_t base, bucket_id_t index_mask) {
        meta.insert_base = base;
        meta.index_mask = index_mask;
    }

    table priority_table { // From prioirty to an insert base and a mask.
        key = {
            hdr.my_protocol.priority : exact;
        }

        actions = {
            set_base;
            NoAction;
        }

        size = 8;
        const default_action = NoAction;
    }

    ////////////////////////////////// sumax a //////////////////////////////////
    Register<sumax, bucket_id_t>(NUMBER_OF_BUCKETS) sumax_l1_a;
    RegisterAction<sumax, bucket_id_t, bit<16>>(sumax_l1_a) apply_sumax_l1_a = {
        void apply(inout sumax reg_data, out bit<16> new_w) {
            if(reg_data.sum_value < meta.sum_w){
                reg_data.sum_value = reg_data.sum_value + 1;
            }
            if(reg_data.max_value < hdr.ipv4.total_len){
                reg_data.max_value = hdr.ipv4.total_len;
            }
            new_w = reg_data.sum_value;
        }
    };
    
    action do_sumax_l1_a() {
        meta.sum_w = apply_sumax_l1_a.execute((bucket_id_t)(meta.index_1));
    }

    Register<sumax, bucket_id_t>(NUMBER_OF_BUCKETS) sumax_l2_a;
    RegisterAction<sumax, bucket_id_t, bit<16>>(sumax_l2_a) apply_sumax_l2_a = {
        void apply(inout sumax reg_data, out bit<16> new_w) {
            if(reg_data.sum_value < meta.sum_w){
                reg_data.sum_value = reg_data.sum_value + 1;
            }
            if(reg_data.max_value < hdr.ipv4.total_len){
                reg_data.max_value = hdr.ipv4.total_len;
            }
            new_w = reg_data.sum_value;
        }
    };
    
    action do_sumax_l2_a() {
        meta.sum_w = apply_sumax_l2_a.execute((bucket_id_t)(meta.index_2));
    }

    Register<sumax, bucket_id_t>(NUMBER_OF_BUCKETS) sumax_l3_a;
    RegisterAction<sumax, bucket_id_t, bit<16>>(sumax_l3_a) apply_sumax_l3_a = {
        void apply(inout sumax reg_data, out bit<16> new_w) {
            if(reg_data.sum_value < meta.sum_w){
                reg_data.sum_value = reg_data.sum_value + 1;
            }
            if(reg_data.max_value < hdr.ipv4.total_len){
                reg_data.max_value = hdr.ipv4.total_len;
            }
            new_w = reg_data.sum_value;
        }
    };
    
    action do_sumax_l3_a() {
        meta.sum_w = apply_sumax_l3_a.execute((bucket_id_t)(meta.index_3));
    }

    ////////////////////////////////// sumax b //////////////////////////////////
    Register<sumax, bucket_id_t>(NUMBER_OF_BUCKETS) sumax_l1_b;
    RegisterAction<sumax, bucket_id_t, bit<16>>(sumax_l1_b) apply_sumax_l1_b = {
        void apply(inout sumax reg_data, out bit<16> new_w) {
            if(reg_data.sum_value < meta.sum_w){
                reg_data.sum_value = reg_data.sum_value + 1;
            }
            if(reg_data.max_value < hdr.ipv4.total_len){
                reg_data.max_value = hdr.ipv4.total_len;
            }
            new_w = reg_data.sum_value;
        }
    };
    
    action do_sumax_l1_b() {
        meta.sum_w = apply_sumax_l1_b.execute((bucket_id_t)(meta.index_1));
    }

    Register<sumax, bucket_id_t>(NUMBER_OF_BUCKETS) sumax_l2_b;
    RegisterAction<sumax, bucket_id_t, bit<16>>(sumax_l2_b) apply_sumax_l2_b = {
        void apply(inout sumax reg_data, out bit<16> new_w) {
            if(reg_data.sum_value < meta.sum_w){
                reg_data.sum_value = reg_data.sum_value + 1;
            }
            if(reg_data.max_value < hdr.ipv4.total_len){
                reg_data.max_value = hdr.ipv4.total_len;
            }
            new_w = reg_data.sum_value;
        }
    };
    
    action do_sumax_l2_b() {
        meta.sum_w = apply_sumax_l2_b.execute((bucket_id_t)(meta.index_2));
    }

    Register<sumax, bucket_id_t>(NUMBER_OF_BUCKETS) sumax_l3_b;
    RegisterAction<sumax, bucket_id_t, bit<16>>(sumax_l3_b) apply_sumax_l3_b = {
        void apply(inout sumax reg_data, out bit<16> new_w) {
            if(reg_data.sum_value < meta.sum_w){
                reg_data.sum_value = reg_data.sum_value + 1;
            }
            if(reg_data.max_value < hdr.ipv4.total_len){
                reg_data.max_value = hdr.ipv4.total_len;
            }
            new_w = reg_data.sum_value;
        }
    };
    
    action do_sumax_l3_b() {
        meta.sum_w = apply_sumax_l3_b.execute((bucket_id_t)(meta.index_3));
    }

    ////////////////////////////////// HH //////////////////////////////////
    // allocate id
    Register<bit<16>,bit<16>>(0x1)cmp_thres;
    RegisterAction<bit<16>,bit<16>,bit<16>>(cmp_thres)HH_identify=
    {
        void apply(inout bit<16> register_data, out bit<16> result){
            if(register_data > 65534)
                register_data = 1;
            else{
                register_data = register_data + 1;
            }
            result = register_data;
        }
    };
    bit<16> HH_index;
    action cmp_HH(){
        HH_index = HH_identify.execute(0);
    }

    // match
    table HH_match_t{
        actions = {
            cmp_HH;
            @defaultonly NoAction;
        }
        key = {
            meta.sum_w:   exact;
        }
        size = 100;
        default_action = NoAction();
        const entries = {
            1100: cmp_HH();
            1200: cmp_HH();
            1300: cmp_HH();
        }
    }

    // insert A
    Register<id_pair,bit<16>>(0x10000)HH_reg_a;
    RegisterAction<id_pair,bit<16>,bit<16>>(HH_reg_a) HH_reg_insert_a=
    {
        void apply(inout id_pair register_data, out bit<16> result){
            register_data.id = hdr.ipv4.dst_addr;
            register_data.counter =(bit<32>)meta.sum_w;
            result = 1;
        }
        
    };

    action HH_insert_a(){
        HH_reg_insert_a.execute(HH_index);
    }

    // insert B
    Register<id_pair,bit<16>>(0x10000)HH_reg_b;
    RegisterAction<id_pair,bit<16>,bit<16>>(HH_reg_b) HH_reg_insert_b=
    {
        void apply(inout id_pair register_data, out bit<16> result){
            register_data.id = hdr.ipv4.dst_addr;
            register_data.counter =(bit<32>)meta.sum_w;
            result = 1;
        }
        
    };

    action HH_insert_b(){
        HH_reg_insert_b.execute(HH_index);
    }

    ////////////////////////////////// send //////////////////////////////////
    action send(bit<9> port) {
        ig_tm_md.ucast_egress_port = port;
    }

    action drop() {
        ig_dprsr_md.drop_ctl = 1;
    }

    table send_table {
        key = {
            ig_intr_md.ingress_port : exact;
        }

        actions = {
            send;
            drop;
            NoAction;
        }

        size = 2;
        const default_action = NoAction;
    }

    ////////////////////////////////// apply //////////////////////////////////
    apply {

        part_switch_table.apply();
        send_table.apply();

        if(hdr.my_protocol.isValid()) {

            apply_hash_1();

            lb_table.apply();
            if(meta.do_insert == 1) {
                priority_table.apply();
                apply_hash_2();
                apply_hash_3();
                bit<32> mask = (bit<32>)(meta.index_mask);
                bit<32> base = (bit<32>)(meta.insert_base);
                meta.index_1 = meta.index_1 & mask;
                meta.index_2 = meta.index_2 & mask;
                meta.index_3 = meta.index_3 & mask;
                meta.index_1 = meta.index_1 + base;
                meta.index_2 = meta.index_2 + base;
                meta.index_3 = meta.index_3 + base;

                if(meta.part_used == 0) {
                    do_sumax_l1_a();
                    do_sumax_l2_a();
                    do_sumax_l3_a();
                }
                else {
                    do_sumax_l1_b();
                    do_sumax_l2_b();
                    do_sumax_l3_b();
                }

                HH_index = 0;
                HH_match_t.apply();
                if(HH_index > 0) {
                    if(meta.part_used == 0)
                        HH_insert_a();
                    else
                        HH_insert_b();
                }
            }
        }
    }
}

control IngressDeparser(packet_out                   pkt,
    /* User */
    inout my_ingress_header_t                        hdr,
    in    my_ingress_metadata_t                      meta,
    /* Intrinsic */
    in    ingress_intrinsic_metadata_for_deparser_t  ig_dprsr_md)
{
    apply {
        pkt.emit(hdr);   
    }
}


struct my_egress_headers_t {

}

struct my_egress_metadata_t {

}

parser EgressParser(packet_in        pkt,
    /* User */
    out my_egress_headers_t          hdr,
    out my_egress_metadata_t         meta,
    /* Intrinsic */
    out egress_intrinsic_metadata_t  eg_intr_md)
{
    /* This is a mandatory state, required by Tofino Architecture */
    state start {
        pkt.extract(eg_intr_md);
        transition accept;
    }
}

control Egress(
    /* User */
    inout my_egress_headers_t                          hdr,
    inout my_egress_metadata_t                         meta,
    /* Intrinsic */    
    in    egress_intrinsic_metadata_t                  eg_intr_md,
    in    egress_intrinsic_metadata_from_parser_t      eg_prsr_md,
    inout egress_intrinsic_metadata_for_deparser_t     eg_dprsr_md,
    inout egress_intrinsic_metadata_for_output_port_t  eg_oport_md)
{
    apply { 

    }
}

control EgressDeparser(packet_out pkt,
    /* User */
    inout my_egress_headers_t                       hdr,
    in    my_egress_metadata_t                      meta,
    /* Intrinsic */
    in    egress_intrinsic_metadata_for_deparser_t  eg_dprsr_md)
{
    apply {
        pkt.emit(hdr);
    }
}

Pipeline(
    IngressParser(),
    Ingress(),
    IngressDeparser(),
    EgressParser(),
    Egress(),
    EgressDeparser()
) pipe;

Switch(pipe) main;

#endif
# EarlyBird – Normalised System Requirements

Source: *Case Study Early Bird Requirements V1.50*. This file turns the 44 original statements into
**atomic system requirements** for embedding. The meaning stays the same. Only the wording changes:

- **Active voice, system as subject:** "The system …"
- **Actor removed / replaced:** per #32 and #34 the phone clerk goes away. Phone interactions become
  customer self-service in the web application, and the process itself stays the same.
- **Manual tools replaced:** text processing system and spreadsheet → the system (#32).
- **Atomic:** compound statements are split into `a`, `b`, `c` …
- **Self-contained:** each sentence names its domain object, so it can be understood without context.
- **Kind:** `FR` = functional, `DR` = domain/data rule (both get embedded), `NFR` = non-functional,
  `CTX` = context only (neither gets embedded).

---

## Company Products and Services

| ID | Kind | Original | Normalised (embedding text) |
|---|---|---|---|
| 1 | NFR | We guarantee breakfast delivery in less than 25 minutes to all parts of the city. | *(excluded – service level)* The breakfast delivery to any address in the city takes less than 25 minutes. |
| 2 | DR | We offer a number of prepackaged breakfasts like the mini-breakfast (…), the luxury breakfast (…) and many more. | The product catalog offers prepackaged breakfast products, such as the mini-breakfast and the luxury breakfast. |
| 3 | FR | Customers can also assemble an individual breakfast to their taste by choosing from a long list of simple products. | The system lets a customer assemble an individual breakfast by selecting simple products from the product catalog. |
| 4 | DR | A prepackaged product may contain simple products, but also other prepackaged products. | A prepackaged product is composed of simple products and/or other prepackaged products (nested product composition). |
| 5 | DR | A typical order consists of various amounts of various products (prepackaged ones and/or simple ones). | An order consists of order lines, each specifying a product (simple or prepackaged) and an amount. |
| 6 | DR | Each product has a unit (e.g. 100 grams) and a price per unit denominated in Euros. | Each product has a unit of measure (e.g. 100 grams) and a price per unit in Euros. |

## Placing an Order

| ID | Kind | Original | Normalised (embedding text) |
|---|---|---|---|
| 7 | CTX | Currently, our customers can place orders only over the phone. | *(excluded – current state; replaced by #32)* |
| 8 | FR | They call our company number 384 29 734 and tell us their customer number. | The system identifies the customer by the customer number when the customer starts placing an order. |
| 9 | DR | A customer number has seven digits preceded by a two-digit area code and followed by a one-digit checksum. | The system validates that a customer number consists of a two-digit area code, seven digits and a one-digit checksum. |
| 10 | DR | There are no collective orders of several customers. | An order always belongs to exactly one customer; the system does not support collective orders of several customers. |
| 11a | FR | After the phone clerk has authenticated the customer … | The system authenticates the customer before an order can be placed. |
| 11b | DR | … (which includes checking that he is not blacklisted e.g. due to bad payment morale) … | During authentication the system checks that the customer is not blacklisted (e.g. due to bad payment morale) and rejects blacklisted customers. |
| 11c | FR | … the customer names one or several products to be added to the shopping cart. | The system lets the customer add one or several products directly to the shopping cart. |
| 12 | FR | If the customer is not sure about certain products, she can tell the desired characteristics (e.g. less than 300 calories and less than € 4) and gets a choice of suitable products. | The system lets the customer search for products by desired characteristics (e.g. less than 300 calories and less than € 4), shows the matching products and lets her add them to the shopping cart. |
| 13 | FR | The customer can also place a follow-up order by naming one of her previous orders, which serves as a blueprint for the new order. | The system lets the customer create a follow-up order by selecting one of her previous orders as a blueprint, copying its products into the shopping cart. |
| 14 | FR | Those three methods of assembling the shopping cart (…) can be combined within placing one order. | The system lets the customer combine direct product selection, product search and a blueprint order when assembling the shopping cart of one order. |
| 15 | DR | An order can only have one blueprint order at most. | An order references at most one blueprint order. |
| 16 | DR | One order can serve as a blueprint many times. | A previous order can be used as the blueprint for any number of new orders. |
| 17a | DR | For each customer, only one address is predefined. Therefore, there is no need to specify a delivery address when ordering. | Each customer has exactly one predefined address, which the system uses automatically as the delivery address of her orders. |
| 17b | FR | The process ends by informing the customer about the order number, which is an eight-digit number with a checksum. | When an order is placed, the system assigns an order number with eight digits and a checksum and shows it to the customer. |

## Packaging an Order

| ID | Kind | Original | Normalised (embedding text) |
|---|---|---|---|
| 18 | FR | A packing clerk looks at each order, assembles the ordered products and puts them into a paper bag. | The system shows the packing clerk each placed order so that she can assemble the ordered products into a paper bag. |
| 19a | FR | The packing clerk prints a label with his own name, the customer's first name and surname, the address, the order number … | The system prints a label for the order bag showing the packing clerk's name, the customer's first name, surname and address, and the order number. |
| 19b | FR | … and the delivery clerk he assigned to the order. | The system lets the packing clerk assign a delivery clerk to the order, and the label shows the assigned delivery clerk. |
| 20 | CTX | The label is attached to the paper bag. | *(excluded – manual physical step)* |
| 21 | FR | The packing clerk prints an invoice twice with the text processing system. | The system prints the invoice of an order in two copies. |
| 22 | FR | The invoice shows the same data as the label plus the individual ordered products, their amounts and prices and the sum total. | The invoice shows the label data plus each ordered product with its amount and price, and the total sum of the order. |
| 23 | DR | If an invoice is printed several times, each copy gets a separate number. | Each printed copy of an invoice gets its own invoice number. |

## Delivering an Order

| ID | Kind | Original | Normalised (embedding text) |
|---|---|---|---|
| 24 | FR | The delivery clerk calculates the optimal itinerary … many orders may be fulfilled within one itinerary. | The system calculates the optimal delivery itinerary for a delivery clerk, covering several orders in one itinerary. |
| 25 | FR | The delivery clerk prints out the optimal itinerary, takes the corresponding bags and invoices and drives to the customers. | The system prints the optimal delivery itinerary together with the list of orders (bags and invoices) the delivery clerk has to deliver. |
| 26 | FR | Each customer confirms delivery by signing a copy of the invoice. She keeps the other copy. | The customer confirms receipt of the delivered order; one invoice copy stays with the customer as her receipt. |

## Status Inquiry

| ID | Kind | Original | Normalised (embedding text) |
|---|---|---|---|
| 27 | FR | Each customer can inquire into the status of her order using the order number. The phone clerk tells whether the delivery clerk is already on his way or not. | The system lets the customer look up the status of her order by order number and shows whether the order is already out for delivery. |

## Canceling an Order

| ID | Kind | Original | Normalised (embedding text) |
|---|---|---|---|
| 28 | FR | Canceling an order is possible by calling the phone clerk and naming the order number. | The system lets the customer cancel an order by its order number. |
| 29 | DR | Once the order has been assembled, canceling is not possible anymore. | The system rejects the cancellation of an order that has already been assembled by the packing clerk. |
| 30 | DR | A cancelation cannot be undone. | A cancelled order cannot be reactivated; cancellation is final. |
| 31 | DR | Updating of orders is not possible. If the customer wants to change an order, she has to cancel it and place a new order. | The system does not allow changes to a placed order; to change it, the customer cancels the order and places a new one. |

## Automation of Existing Processes

| ID | Kind | Original | Normalised (embedding text) |
|---|---|---|---|
| 32 | CTX | We want the current process to be automated (but not changed) by a web-based application. It replaces ordering by phone, the text processing system and the itinerary tools. | *(excluded – scope statement; already applied to all requirements above)* |
| 33 | NFR | The application is supposed to support the following browsers in the following versions: … | *(excluded – technical constraint)* |
| 34a | FR | The user groups are: customers, packing clerks, delivery clerks and managers. | The system authorises users according to their role: customer, packing clerk, delivery clerk or manager. |
| 34b | CTX | There will be no phone clerks in the future. | *(excluded – already applied)* |
| 35 | FR | The confirmation of the delivery should be browser-based: the delivery clerk presents a smartphone to the customer who enters her password in a browser window to confirm the delivery. | The system lets the customer confirm the delivery of her order by entering her password in a browser on the delivery clerk's smartphone. |
| 36 | FR | The software should offer a browser-based product search function that requires no authentication. It replaces having the phone clerk list suitable products. | The system offers a public product search that anyone can use without logging in. |

## Placing and Canceling an Order by Text Message

| ID | Kind | Original | Normalised (embedding text) |
|---|---|---|---|
| 37a | FR | Customers can also order by text messages (SMS), sending a defined string to our company number. | The system accepts orders sent by customers as SMS text messages to the company number. |
| 37b | DR | The string looks like `C 2339110045 W mypswd P EGG 1 P TOAST 2 P OJUICE 1` (customer number, password, products with amounts). | The system parses an SMS order message containing the customer number (C), the password (W) and one or more product codes with amounts (P). |
| 38 | DR | Each product has a unique product code. | Each product has a unique product code. |
| 39 | DR | The string may contain any number of products but is limited by the maximum SMS length. | An SMS order may contain any number of products, limited only by the maximum length of a text message. |
| 40 | FR | The system responds with a text message specifying the order number. | The system replies to an SMS order with a text message containing the assigned order number. |
| 41 | FR | Canceling by text message is possible by sending `O 261273842`. | The system cancels an order when it receives an SMS cancellation message containing the order number (O). |

## Payment

| ID | Kind | Original | Normalised (embedding text) |
|---|---|---|---|
| 42 | DR | We have a payment system already in place. It receives interface records for expected payments: customerNumber, orderNumber, amountInEuros and expected payment date. | An expected-payment record for the external payment system contains the customer number, the order number, the amount in Euros and the expected payment date. |
| 43 | FR | Once the packing clerk has finished assembling an order, a record representing an expected payment should be transferred to the payment system. | When the packing clerk has finished assembling an order, the system sends an expected-payment record to the external payment system. |

## Business Reporting

| ID | Kind | Original | Normalised (embedding text) |
|---|---|---|---|
| 44 | FR | The system prints a business report for the manager every night automatically, listing the orders of a given day with product, amount, packing clerk, delivery clerk, customer, address and order number. | Every night the system automatically prints a business report for the manager listing all orders of the day with products, amounts, packing clerk, delivery clerk, customer, customer address and order number. |

---

## Summary

| Kind | Count | Embedded |
|---|---|---|
| FR | 26 | ✅ |
| DR | 18 | ✅ |
| NFR | 2 (1, 33) | ❌ |
| CTX | 4 (7, 20, 32, 34b) | ❌ |

**44 items get embedded** (counting split sub-requirements).

### Notes on interpretation (please review)
- **#8/#11a:** the phone call becomes customer identification and authentication in the web app.
  The password comes from #35/#37, so the password exists in the domain.
- **#18/#19b:** the packing clerk keeps assigning the delivery clerk. The process is unchanged (#32).
- **#26 vs #35:** #35 replaces the signature with a password confirmation. #26 is kept for the
  "customer keeps one invoice copy" part, so the two may overlap semantically. That is intended.
- **#36 vs #12:** both stay. #12 is search *while ordering*, #36 is *public, unauthenticated* search.
